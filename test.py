from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import json
import sys
import io
import re
import subprocess
import signal
from contextlib import contextmanager


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
    # first try: triple-backtick block preceded by "solution:"
    pattern_solution = r"(?i)solution:\s*```(?:\w+\n)?(.*?)```"
    match = re.search(pattern_solution, response, re.DOTALL)
    if match:
        return match.group(1)

    # fallback: first triple-backtick block regardless of preceding text
    pattern_fallback = r"```(?:\w+\n)?(.*?)```"
    match = re.search(pattern_fallback, response, re.DOTALL)
    if match:
        return match.group(1)

    return None


def test_problem(problem, model_name):
    print("beginning problem ", problem["id"], "\n\n")
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

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=model_name)
    chain = prompt | model

    input_vars = {"question": problem["statement"]}

    print("model responding")
    try:
        with time_limit(900):
            llm_response = chain.invoke(input_vars)
        print("model responded")
    except TimeoutException:
        print("model timed out")
        return "model_timeout"

    output = io.StringIO()
    sys.stdout = output

    namespace = {}
    error_condition = None

    llm_code = extract_code(llm_response)

    if llm_code:
        try:
            install_missing_packages(llm_code)
            try:
                with time_limit(60):
                    exec(llm_code, namespace)

                    if "solution" in namespace:
                        result = namespace["solution"]()
                    else:
                        print("no solution() function found in model's response")
                        error_condition = "solution_not_found"

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

model_name = "llama3.2:1b"


def run_benchmark(model_name):
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
    }

    with open("problems.json", "r") as f:
        problems = {p["id"]: p for p in json.load(f)}

    with open("easiest_100.json", "r") as f:
        easiest_problem_ids = json.load(f)

    for problem_id in easiest_problem_ids[:5]:
        if problem_id in problems:
            problem = problems[problem_id]
            result = test_problem(problem, model_name)
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
        "wins": results["wins"],
        "fails": results["fails"],
        "model_timeout": results["model_timeout"],
        "solution_not_found": results["solution_not_found"],
        "code_timeout_fail": results["code_timeout_fail"],
        "code_exec_fail": results["code_exec_fail"],
        "package_install_fail": results["package_install_fail"],
        "regex_match_fail": results["regex_match_fail"],
    }

    print("Results:", new_result)
    existing_results.append(new_result)

    with open("results-100-1shot.json", "w") as f:
        json.dump(existing_results, f, indent=2)

    print("were done")


run_benchmark(model_name)
