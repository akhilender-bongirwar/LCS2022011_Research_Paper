import os

from dotenv import load_dotenv
load_dotenv()

import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
import numpy as np

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Pydantic models (reusing LoT structures)
class LogicalProposition(BaseModel):
    symbol: str
    description: str

class LogicalExpression(BaseModel):
    expression: str
    description: str

class LogicOfThoughtOutput(BaseModel):
    propositions: List[LogicalProposition] = Field(..., description="List of logical propositions")
    expressions: List[LogicalExpression] = Field(..., description="List of logical expressions")
    extended_expressions: List[LogicalExpression] = Field(..., description="List of extended logical expressions")
    translated_description: str = Field(..., description="Natural language description of the logical reasoning")
    solution: str = Field(..., description="Solution to the puzzle")

# --- Adaptive Logic-of-Thought Modules ---

def assess_complexity(context: str, llm_model="gpt-3.5-turbo") -> str:
    """
    [A-LoT Module]
    Assesses the complexity of the input context using an LLM.
    """
    prompt = f"""Analyze the following reasoning problem and classify its complexity as 'low', 'medium', or 'high'.
    Consider the number of entities, relationships, logical steps required, and potential for ambiguity.
    Provide your classification and a brief justification.

    Problem:
    {context}

    Complexity Classification (low/medium/high):
    Justification:"""
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are an expert in assessing the complexity of logical reasoning problems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        classification_output = response.choices[0].message.content.strip().lower()
        if "high" in classification_output:
            return "high"
        elif "medium" in classification_output:
            return "medium"
        else:
            return "low"
    except Exception as e:
        print(f"Error in complexity assessment: {e}")
        return "medium"  # Default to medium in case of error

def analyze_context(context: str, llm_model="gpt-3.5-turbo") -> Dict[str, float]:
    """
    [A-LoT Module]
    Analyzes the context using an LLM to identify the relevance of different logical aspects.
    Returns a dictionary with scores (0.0 to 1.0) indicating relevance.
    """
    prompt = f"""Analyze the following reasoning problem and estimate the relevance (on a scale of 0.0 to 1.0) of the following logical aspects for solving it:
    - Propositional Logic (e.g., AND, OR, NOT, IF-THEN)
    - Relational Reasoning (relationships between entities)
    - Numerical Reasoning (if numbers and calculations are involved)
    - Temporal Reasoning (if time or sequences are important)

    Provide your response as a JSON object where keys are the logical aspects and values are their relevance scores.

    Problem:
    {context}

    {{
      "propositional_logic": 0.0,
      "relational_reasoning": 0.0,
      "numerical_reasoning": 0.0,
      "temporal_reasoning": 0.0
    }}"""
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing the logical structure of reasoning problems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in context analysis: {e}")
        return {"propositional_logic": 0.5, "relational_reasoning": 0.5, "numerical_reasoning": 0.0, "temporal_reasoning": 0.0} # Default

def adaptive_logic_augmentation(context: str, complexity: str, context_relevance: Dict[str, float], llm_model="gpt-4-turbo-preview") -> str:
    """
    [A-LoT Module]
    Dynamically adjusts the logical augmentation based on complexity and context relevance.
    """
    if complexity == "low":
        return context  # No explicit logic injection for very simple cases
    elif complexity == "medium":
        prompt = f"""Consider the following reasoning problem. Based on its nature and the relevance of different logical aspects (Propositional: {context_relevance.get('propositional_logic', 0.5)}, Relational: {context_relevance.get('relational_reasoning', 0.5)}), extract a concise set of the most important logical constraints or relationships that would aid in solving it.

        Problem:
        {context}

        Concise Logical Constraints:"""
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert in extracting key logical constraints from reasoning problems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            logical_hints = response.choices[0].message.content.strip()
            return f"{context}\n\nLogical Hints: {logical_hints}"
        except Exception as e:
            print(f"Error in moderate augmentation: {e}")
            return context
    elif complexity == "high":
        prompt = f"""Extract detailed logical propositions and expressions from the following context, focusing on capturing all key relationships and constraints. Provide the output as a JSON object with 'propositions' and 'expressions' keys. Each proposition should have 'symbol' and 'description', and each expression should have 'expression' and 'description'.

        Context:
        {context}

        JSON Output:"""
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert in logical reasoning and propositional logic."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            return f"{context}\n\nLogical Augmentation: {response.choices[0].message.content}"
        except Exception as e:
            print(f"Error in high augmentation: {e}")
            return context
    return context

# --- Reusing LoT functions with potential adaptation in prompting ---

def logic_extension(expressions: List[LogicalExpression], llm_model="gpt-4-turbo-preview") -> List[LogicalExpression]:
    """
    Implements the Logic Extension phase of LoT. Can be adapted with more sophisticated prompting.
    """
    prompt = f"""Extend the following logical expressions using logical reasoning laws to derive new, relevant inferences that could help solve the problem. Provide the output as a JSON array of extended expressions with "extended_expressions" root key. Each expression should be an object with 'expression' and 'description' keys.

    Logical Expressions:
    {json.dumps([expr.model_dump() for expr in expressions])}

    JSON Output:"""
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are an expert in logical reasoning and propositional logic."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        result = json.loads(response.choices[0].message.content)
        return [LogicalExpression(**expr) for expr in result.get('extended_expressions', [])]
    except Exception as e:
        print(f"Error in logic extension: {e}")
        return []

def logic_translation(extended_expressions: List[LogicalExpression], llm_model="gpt-4-turbo-preview") -> str:
    """
    Implements the Logic Translation phase of LoT.
    """
    prompt = f"""Translate the following extended logical expressions into a natural language description of the logical reasoning process.

    Extended Logical Expressions:
    {json.dumps([expr.model_dump() for expr in extended_expressions])}

    Natural Language Description:"""
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are an expert in explaining logical reasoning in natural language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in logic translation: {e}")
        return ""

def solve_puzzle(context: str, augmented_context: str, translated_description: str, complexity: str, llm_model="gpt-4-turbo-preview") -> str:
    """
    Uses the original context, augmented context, and translated logical reasoning to solve the puzzle.
    Adapts the prompting based on complexity.
    """
    if complexity == "low":
        prompt = f"""Solve the following puzzle using chain of thought:
        {context}
        Provide a step-by-step explanation and the final answer."""
    else:
        prompt = f"""Using the following context, the adaptively injected logical information, and the derived logical reasoning:

        Original Context:
        {context}

        Adaptively Injected Logic:
        {augmented_context.split('Logical Augmentation:')[-1] if 'Logical Augmentation:' in augmented_context else (augmented_context.split('Logical Hints:')[-1] if 'Logical Hints:' in augmented_context else 'None')}

        Logical Reasoning:
        {translated_description}

        Solve the puzzle: {context.split('Who will')[0].strip()}
        Provide a step-by-step explanation of your reasoning, and then state the final answer."""
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are an expert puzzle solver using logical reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in solving puzzle: {e}")
        return "Could not solve the puzzle."

def run_adaptive_logic_of_thought(context: str) -> LogicOfThoughtOutput:
    """
    Orchestrates the entire Adaptive Logic-of-Thought process.
    """
    # 1. Complexity Assessment
    complexity = assess_complexity(context)
    print(f"Input Complexity: {complexity}")

    # 2. Contextual Analysis
    context_relevance = analyze_context(context)
    print(f"Context Relevance: {context_relevance}")

    # 3. Adaptive Logic Augmentation
    augmented_context = adaptive_logic_augmentation(context, complexity, context_relevance)
    print(f"Augmented Context:\n{augmented_context}")

    propositions = []
    expressions = []
    extended_expressions = []
    translated_description = ""
    solution = ""

    # 4. Logic Processing for Medium and High Complexity
    if complexity in ["medium", "high"]:
        # Extract Logic
        logic_extraction_prompt = augmented_context.split('Logical Augmentation:')[-1] if 'Logical Augmentation:' in augmented_context else (augmented_context.split('Logical Hints:')[-1] if 'Logical Hints:' in augmented_context else context)
        try:
            extraction_result = json.loads(client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert in logical reasoning and propositional logic."},
                    {"role": "user", "content": f"""Extract logical propositions and expressions from: {logic_extraction_prompt}\n\nProvide as JSON with 'propositions' and 'expressions'."""}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            ).choices[0].message.content)
            propositions = [LogicalProposition(**prop) for prop in extraction_result.get('propositions', [])]
            expressions = [LogicalExpression(**expr) for expr in extraction_result.get('expressions', [])]

            # Extend Logic
            extended_expressions = logic_extension(expressions)

            # Translate Logic
            translated_description = logic_translation(extended_expressions)

            # Solve Puzzle
            solution = solve_puzzle(context, augmented_context, translated_description, complexity)

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError during logic processing: {e}")
            solution = solve_puzzle(context, augmented_context, "", complexity) # Try solving without full logic
        except Exception as e:
            print(f"Error during logic processing: {e}")
            solution = solve_puzzle(context, augmented_context, "", complexity) # Try solving without full logic

    # 5. Direct Solving for Low Complexity
    else:
        prompt_simple_solve = f"""Solve the following puzzle using chain of thought:
        {context}
        Provide a step-by-step explanation and the final answer."""
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert puzzle solver using logical reasoning."},
                    {"role": "user", "content": prompt_simple_solve}
                ],
                temperature=0.3,
            )
            solution = response.choices[0].message.content.strip()
            translated_description = "Solved with standard Chain of Thought due to low complexity."
        except Exception as e:
            print(f"Error during simple solving: {e}")
            solution = "Could not solve the puzzle."

    return LogicOfThoughtOutput(
        propositions=propositions,
        expressions=expressions,
        extended_expressions=extended_expressions,
        translated_description=translated_description,
        solution=solution
    )

# Example puzzles
einsteins_riddle = """
The Simpsons are preparing a concert show with tricks, a guitar song, and one family member's own poem.
A person who wrote the poem will give a bread machine as a present and buy irises.
Mummy will buy tulips.
Melanie has learned to bake cinnamon buns and remembered guitar chords for Granny's birthday.
The trickster has prepared a notebook for recipes and a fruit salad.
Melanie knows that Granny likes daisies.
A person who will give a rocking chair will also prepare homemade candies for Granny.
Bill has a special deck of cards and a box with a double bottom.
Mummy and Melanie have had rehearsals for two for a week.
Daddy will prepare orange juice.
Granny will also be given a bouquet of roses and a ticket to the theatre play.
Who will give a theatre ticket and who will buy roses?
"""

simple_arithmetic = "What is the result of 15 multiplied by 3?"

simple_logic = "If it is raining, then the ground is wet. It is raining. Is the ground wet?"

def main():
    print("\nSolving Einstein's Riddle using Adaptive Logic-of-Thought:")
    result_complex = run_adaptive_logic_of_thought(einsteins_riddle)
    print(f"\nA-LoT Output (Complex):\n{result_complex.solution}\nReasoning:\n{result_complex.translated_description}")

    print("\nSolving Simple Arithmetic using Adaptive Logic-of-Thought:")
    result_simple = run_adaptive_logic_of_thought(simple_arithmetic)
    print(f"\nA-LoT Output (Simple):\n{result_simple.solution}\nReasoning:\n{result_simple.translated_description}")

    print("\nSolving Simple Logic using Adaptive Logic-of-Thought:")
    result_logic = run_adaptive_logic_of_thought(simple_logic)
    print(f"\nA-LoT Output (Simple Logic):\n{result_logic.solution}\nReasoning:\n{result_logic.translated_description}")

if __name__ == "__main__":
    main()