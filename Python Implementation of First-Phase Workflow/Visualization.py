import matplotlib.pyplot as plt

# Compare classification performance
performance = {
    "Training": train_accuracy * 100,
    "Validation": test_accuracy * 100
}
plt.bar(performance.keys(), performance.values(), color='skyblue')
plt.ylabel("Accuracy (%)")
plt.title("Classification Performance")
plt.show()
