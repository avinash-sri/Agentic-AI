from graph import app

result = app.invoke({
    "task": "Write a Python function that calculates the average of a list of numbers, and handles empty lists, non-numeric values, and returns the result rounded to 2 decimal places.",
    "plan": "", "code": "", "review": "", "approved": False
})

print("PLAN:\n", result["plan"])
print("\nFINAL CODE:\n", result["code"])
print("\nREVIEW:\n", result["review"])