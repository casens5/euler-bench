from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
import json
import sys
import io
import re
import subprocess
import signal
import dotenv
from contextlib import contextmanager
import psutil
import os
import threading
import time


dotenv.load_dotenv()


def install_missing_packages(code: str):
    """
    Scan the code for import statements, determine if each package is installed,
    and install it via pip if not.
    """
    # Regex patterns for import lines:
    # - Pattern 1 matches: import package[.subpackage] [as alias][, package2 [as alias2], ...]
    # - Pattern 2 matches: from package import something
    pattern_import = (
        r"^\s*import\s+((?:[a-zA-Z0-9_]+(?:\s+as\s+[a-zA-Z0-9_]+)?(?:\s*,\s*)?)+)"
    )
    pattern_from = r"^\s*from\s+([a-zA-Z0-9_]+)\s+import\s+"

    packages_to_install = set()
    for line in code.splitlines():
        # Check "import foo[, bar]"
        match_import = re.match(pattern_import, line)
        if match_import:
            # Split multiple imports and handle 'as' aliases
            imports = match_import.group(1).split(",")
            for imp in imports:
                # Take just the package name, before any 'as' statement
                package = imp.strip().split(" as ")[0].strip()
                packages_to_install.add(package)
            continue

        # Check "from foo import bar"
        match_from = re.match(pattern_from, line)
        if match_from:
            packages_to_install.add(match_from.group(1))

    # Attempt to import each package, install if that fails
    for package in packages_to_install:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package], check=True
            )


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def log_to_file(text: str, filename: str = "log.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{text}\n")


def extract_code(response):
    # Remove text between <think> tags temporarily
    think_blocks = []
    pattern_think = r"<think>(.*?)</think>"
    cleaned_response = re.sub(
        pattern_think,
        lambda m: think_blocks.append(m.group(1)) or "",
        response,
        flags=re.DOTALL,
    )

    # first try: triple-backtick block preceded by "solution:" in non-think text
    pattern_solution = r"(?i)solution:\s*```(?:\w+\n)?(.*?)```"
    match = re.search(pattern_solution, cleaned_response, re.DOTALL)
    if match:
        return match.group(1)

    # second try: first triple-backtick block in non-think text
    pattern_fallback = r"```(?:\w+\n)?(.*?)```"
    match = re.search(pattern_fallback, cleaned_response, re.DOTALL)
    if match:
        return match.group(1)

    # If no matches found, search in the think blocks
    for think_block in think_blocks:
        # Try solution pattern first
        match = re.search(pattern_solution, think_block, re.DOTALL)
        if match:
            return match.group(1)

        # Try fallback pattern
        match = re.search(pattern_fallback, think_block, re.DOTALL)
        if match:
            return match.group(1)

    return None


class ResourceMonitor:
    def __init__(self, pid, max_cpu_percent=90, max_memory_gb=4):
        self.pid = pid
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_gb = max_memory_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        self.should_stop = False
        self.resource_exceeded = False

    def monitor(self):
        try:
            process = psutil.Process(self.pid)
            while not self.should_stop:
                # Check CPU usage
                cpu_percent = process.cpu_percent(interval=0.5)
                # Check memory usage
                memory_info = process.memory_info()

                if (
                    cpu_percent > self.max_cpu_percent
                    or memory_info.rss > self.max_memory_gb
                ):
                    self.resource_exceeded = True
                    process.kill()
                    break
                time.sleep(0.5)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def test_problem(problem, model_name, api="ollama"):
    print("\n\nbeginning problem ", problem["id"])
    template = """
        Solve this math/programming challenge by writing a python script.  You can import any packages you need.  Solutions will be given a maximum of 60 seconds to execute using moderate hardware.  Format your solution like this:

        SOLUTION:
        ```
        def solution():
            # your solution here
            return answer
        ```

        That is, your response must contain `SOLUTION:`, then a code block beginning and ending with three backticks (```).  All python code in this block will be executed (package imports, variable and function definitions, etc), but you must include a `solution` function, which returns the value that the question asks for.

        Question: {question}
    """

    prompt = PromptTemplate.from_template(template)

    if api == "ollama":
        model = OllamaLLM(model=model_name)
    elif api == "openai":
        model = ChatOpenAI(model=model_name)
    else:
        raise ValueError(f"Unsupported API: {api}")

    chain = prompt | model

    input_vars = {"question": problem["statement"]}
    error_condition = None

    print("model responding")
    try:
        with time_limit(300):
            llm_response = chain.invoke(input_vars)
        print("model responded")
    except TimeoutException:
        print("model timed out")
        error_condition = "model_timeout"
        return error_condition

    if api == "openai":
        llm_response = llm_response.content

    output = io.StringIO()
    sys.stdout = output

    namespace = {}

    llm_code = extract_code(llm_response)

    if llm_code:
        try:
            install_missing_packages(llm_code)
            try:
                with time_limit(60):
                    # Start resource monitoring in a separate thread
                    monitor = ResourceMonitor(os.getpid())
                    monitor_thread = threading.Thread(target=monitor.monitor)
                    monitor_thread.start()

                    try:
                        exec(llm_code, namespace)

                        if "solution" in namespace:
                            result = namespace["solution"]()
                        else:
                            print("no solution() function found in model's response")
                            error_condition = "solution_not_found"
                    finally:
                        # Stop the monitoring thread
                        monitor.should_stop = True
                        monitor_thread.join()

                        if monitor.resource_exceeded:
                            print("Solution exceeded resource limits")
                            error_condition = "resource_limit_exceeded"

            except TimeoutException:
                print("Solution timed out after 60 seconds")
                error_condition = "code_timeout_fail"
            except Exception as e:
                print(f"Error in solution execution: {e}")
                error_condition = "code_exec_fail"
        except Exception as e:
            print(f"failed to install packages: {e}")
            error_condition = "package_install_fail"
    else:
        print("no regex match found in model's response")
        error_condition = "regex_match_fail"

    sys.stdout = sys.__stdout__

    log_to_file(
        f"*********\nproblem: {problem['id']}\n\nmodel_response:\n{llm_response}\n....\n\nmatch:\n{llm_code if llm_code else None}\n\n.....\nerror_condition: {error_condition}\n\nsuccess: {str(result) == str(problem['solution']) if 'result' in locals() else None}\n\n",
        "log.txt",
    )

    # print("Captured Output:", output.getvalue())
    # print("Result:", result)
    if error_condition:
        return error_condition
    return str(result) == str(problem["solution"])


with open("problems.json", "r") as f:
    problems = json.load(f)

# model_name = "gpt-4o-mini"
model_name = "llama3.2:1b"
# api = "openai"
api = "ollama"


def run_benchmark(model_name="llama3.2:1b", api="ollama"):
    print("lets go")
    results = {
        "wins": [],
        "fails": [],
        "model_timeout": [],
        "solution_not_found": [],
        "code_timeout_fail": [],
        "code_exec_fail": [],
        "package_install_fail": [],
        "regex_match_fail": [],
        "resource_limit_exceeded": [],  # Add new error category
    }

    with open("problems.json", "r") as f:
        problems = {p["id"]: p for p in json.load(f)}

    with open("medium_300.json", "r") as f:
        medium_problem_ids = json.load(f)

    problem_ids = medium_problem_ids[:100]
    for problem_id in problem_ids:
        if problem_id in problems:
            problem = problems[problem_id]
            result = test_problem(problem, model_name, api)
            if result is True:
                results["wins"].append(problem["id"])
            elif result is False:
                results["fails"].append(problem["id"])
            else:
                results[result].append(problem["id"])
        else:
            print(
                """
                  **********************************
                  id does not exist error!
                  **********************************
                  """
            )

    try:
        with open("results-100-1shot.json", "r") as f:
            existing_results = json.load(f)
    except FileNotFoundError:
        existing_results = []

    new_result = {
        "model": model_name,
        "wins_count": len(results["wins"]),
        "problems_count": len(problem_ids),
        "wins": results["wins"],
        "fails": results["fails"],
        "model_timeout": results["model_timeout"],
        "solution_not_found": results["solution_not_found"],
        "code_timeout_fail": results["code_timeout_fail"],
        "code_exec_fail": results["code_exec_fail"],
        "package_install_fail": results["package_install_fail"],
        "regex_match_fail": results["regex_match_fail"],
        "resource_limit_exceeded": results["resource_limit_exceeded"],
    }

    print("Results:", new_result)
    existing_results.append(new_result)

    with open("results-100-1shot.json", "w") as f:
        json.dump(existing_results, f, indent=2)

    print("were done")


run_benchmark(model_name, api)
