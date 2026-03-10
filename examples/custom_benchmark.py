"""Example: Run a custom benchmark programmatically."""

from autoagentlab import Agent, ExperimentLoop


def main():
    # Define your own tasks as [question, expected_answer] pairs
    my_tasks = [
        ["What is 15 * 13?", "195"],
        ["What is the derivative of x^2?", "2x"],
        ["Convert 100 Fahrenheit to Celsius (round to nearest integer)", "38"],
        ["What is the sum of angles in a triangle?", "180"],
        ["What is 2^10?", "1024"],
    ]

    # Create an agent with a custom prompt
    agent = Agent(
        prompt="You are a math tutor. Answer math questions with just the number or expression.",
        model="gpt-4o-mini",  # or any model you prefer
    )

    # Run the experiment loop
    loop = ExperimentLoop(agent, my_tasks, max_iterations=3)
    history = loop.run()

    # Access results programmatically
    best = max(history, key=lambda r: r.accuracy)
    print(f"\nBest accuracy: {best.accuracy * 100:.0f}% at iteration {best.iteration}")


if __name__ == "__main__":
    main()
