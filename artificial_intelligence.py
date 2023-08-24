from sklearn import tree

# Collect training data
# Features: [weight, texture] where 0 represents "smooth" and 1 represents "rough"
features = [[140, 0], [130, 0], [150, 1], [170, 1]]
# Labels: 0 represents "apple" and 1 represents "orange"
labels = [0, 0, 1, 1]

# Create a decision tree classifier
clf = tree.DecisionTreeClassifier()

# Train the classifier on the training data
clf = clf.fit(features, labels)

# Predict new examples
# Here, we're predicting a fruit with weight 160 and rough texture (1)
prediction = clf.predict([[160, 1]])

# Print the prediction
if prediction == 0:
    print("Predicted: Apple")
else:
    print("Predicted: Orange")
