from dagqa.eval.metrics import answer_f1, exact_match, cosine_sim

while True:
    stringA = input("String A :")
    stringB = input("String B :")
    print("The cosine distance is",cosine_sim(stringA,stringB))