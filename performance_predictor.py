def get_average(marks):
    return sum(marks) / len(marks)

def categorize_performance(average):
    if average >= 75:
        return "Excellent"
    elif 50 <= average < 75:
        return "Average"
    else:
        return "Needs Improvement"

def main():
    print("Welcome to the Student Performance Predictor!")
    subjects = ["Math", "Science", "English"]
    marks = []

    for subject in subjects:
        mark = float(input(f"Enter marks for {subject} (out of 100): "))
        marks.append(mark)

    avg = get_average(marks)
    category = categorize_performance(avg)

    print(f"\nAverage Marks: {avg:.2f}")
    print(f"Performance: {category}")

if __name__ == "__main__":
    main()
