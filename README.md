# Project 2
## Group Members:
1) Ragasree Katam  - A20552861
2) Sesha Sai Sushmith Muchintala  -A20536372
3) Udhay Chander Bharatha  -A20518701

## Boosting Trees

Implement again from first principles the gradient-boosting tree classification algorithm (with the usual fit-predict interface as in Project 1) as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). Answer the questions below as you did for Project 1. In this assignment, you'll be responsible for developing your own test data to ensure that your implementation is satisfactory. (Hint: Use the same directory structure as in Project 1.)

The same "from first principals" rules apply; please don't use SKLearn or any other implementation. Please provide examples in your README that will allow the TAs to run your model code and whatever tests you include. As usual, extra credit may be given for an "above and beyond" effort.

As before, please clone this repo, work on your solution as a fork, and then open a pull request to submit your assignment. *A pull request is required to submit and your project will not be graded without a PR.*



# STEPS TO RUN

Windows:
# Create the virtual environment (Python 3.3+)
python -m venv venv

# Activate the Virtual Environment
.\venv\Scripts\activate

# Install Project Dependencies
pip install -r requirements.txt

# Run The Project
python run.py --n_estimators 100 --lr 0.1 --depth 3

# Run The Tests
python -m pytest -q

# Deactivate
deactivate

# ------------------------------------------------------------------------------------------------------------------------------------

Put your README below. Answer the following questions.

# ------------------------------------------------------------------------------------------------------------------------------------

*  What does the model you have implemented do and when should it be used?

Think of it as a committee of small decision trees.

->The first tree takes a guess at the labels.

->The second tree studies the mistakes the first one made and tries to correct them.

->The third tree fixes what the first two still get wrong, and so on.

After a few hundred rounds the whole committee votes and you get a strong “yes / no” prediction.
Use it when your data live in a spreadsheet-style table (numbers, categories, dates, etc.) and the relationship between columns and the label might be complicated or non-linear. It often beats a single tree or a plain logistic-regression model while staying fairly interpretable.

# ------------------------------------------------------------------------------------------------------------------------------------

* How did you test your model to determine if it is working reasonably correctly?

1. Automated tests — four PyTest cases run every time:

->learns a curved boundary with >90 % accuracy,

->fits a trivial 1-feature dataset perfectly,

->predict_proba returns the right shape and stays between 0-1,

->predict outputs only 0s and 1s.

2. Loss-curve eyeball test — during training we plot train vs. validation log-loss.

->Train loss should go down smoothly.

->Validation loss should drop at first, then flatten or rise slightly (classic “don’t overfit” pattern).
If those lines look sensible the algorithm is behaving.

# ------------------------------------------------------------------------------------------------------------------------------------

* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)



| **Knob**                | **What it controls**                         | **Typical effect on    training**    
|-------------------------|----------------------------------------------|-----------------------------------------------| 
| `n_estimators`          | Number of trees in the ensemble              | ↑ more trees → better fit but longer runtime
| `learning_rate`         | How much each tree can change the prediction | ↓ lower rate → need more trees, usually better generalisation
| `max_depth`             | Depth of each tree                           | ↑ deeper trees capture complex rules but may  over-fit
| `subsample`             | Fraction of rows used per tree               | Adds randomness; reduces over-fitting (like bagging)
| `colsample`             | Fraction of columns considered per split     | Skips some noisy features each split; improves robustness
| `early_stopping_rounds` | Patience on validation loss                  | Stops when val-loss stops improving; saves time, avoids late over-fit


# ------------------------------------------------------------------------------------------------------------------------------------

* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

->Text-like or super-sparse data (tens of thousands of zero-heavy columns). Our simple splitter scans every value, so it slows down.

->Huge datasets (100 k+ rows or 10 k+ features). Python loops aren’t fast enough; a histogram-based or C-accelerated version would be better.

->Extreme class imbalance (e.g. 1 % positives). The default loss treats each row equally, so the rare class may get ignored without extra weighting.

->More than two classes. We only coded the binary log-loss. Multi-class support would need a soft-max extension.

None of these are fundamental roadblocks—they just need extra engineering time that’s beyond this “from scratch” exercise.


# ------------------------------------------------------------------------------------------------------------------------------------