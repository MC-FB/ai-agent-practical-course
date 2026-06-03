# to test, run powershell with docker compose run --rm -v "${PWD}/tests:/app/tests" dagqa python tests/metrics_test.py

from dagqa.eval.metrics import answer_f1, exact_match, cosine_sim


test_predictions = ["Hi","insert sentence here"]
test_gts = ["Hello", "sentence insert here"]

for test_pred,test_gt in zip(test_predictions,test_gts):
    print("Test predicted sentence    : " + test_pred)
    print("Test ground truth sentence : " + test_gt)
    print("The exact match score is       :" + str( exact_match(test_pred,test_gt)))
    print("The f1 term match score is     :" + str( answer_f1(test_pred,test_gt)))
    print("The cosine similarity score is :" + str( cosine_sim(test_pred,test_gt)))
    