import re
import random
import ast
import operator


def _normalize_equation_text(equation_str):
    if equation_str is None:
        return None
    eq = str(equation_str)
    # Normalize common LaTeX/math variants to plain arithmetic operators.
    eq = re.sub(r"\\left|\\right", "", eq)
    eq = re.sub(r"\\times|\\cdot|×", "*", eq)
    eq = re.sub(r"\\div|÷", "/", eq)
    eq = eq.replace("\\(", "(").replace("\\)", ")")
    eq = eq.replace("$", "")
    eq = eq.strip().strip("`")
    eq = re.sub(r"\s+", " ", eq).strip()
    return eq if eq else None


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    #if "Assistant:" in solution_str:
    #    solution_str = solution_str.split("Assistant:", 1)[1]
    #elif "<|im_start|>assistant" in solution_str:
    #    solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    #else:
    #    return None
    #solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, flags=re.IGNORECASE | re.DOTALL)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
        return _normalize_equation_text(final_answer)
    else:
        # Fallback: tolerate truncated generation that has "<answer>" without closing tag.
        open_matches = list(re.finditer(r"<answer>", solution_str, flags=re.IGNORECASE))
        if open_matches:
            start = open_matches[-1].end()
            tail = solution_str[start:]
            # If other tags appear after <answer>, keep text before the next tag.
            next_tag = re.search(r"</?[a-zA-Z_][^>]*>", tail)
            if next_tag:
                tail = tail[: next_tag.start()]
            return _normalize_equation_text(tail.strip())
    return None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        equation_str = _normalize_equation_text(equation_str)
        if equation_str is None:
            return False
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        equation_str = _normalize_equation_text(equation_str)
        if equation_str is None:
            return None
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    do_print = 0
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        # print("evaluate equation countdown")
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score 
