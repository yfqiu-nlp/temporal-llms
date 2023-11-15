import sys
import json

output_path=sys.argv[1]
model_name=sys.argv[2]
    
with open("dataset/mctaco/mctaco.json", "r") as f:
    dataset = json.load(f)

with open(output_path+"/mctaco-test-"+model_name+".out") as f:
    predictions = [d for d in f.readlines() if d != ""]

letter_index = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(N)', '(M)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']

# letter_index_b = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 'I)', 'J)', 'K)', 'L)', 'N)', 'M)', 'O)', 'P)', 'Q)', 'R)', 'S)', 'T)', 'U)', 'V)', 'W)', 'X)', 'Y)', 'Z)']

# letter_index_c = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.', 'J.', 'K.', 'L.', 'N.', 'M.', 'O.', 'P.', 'Q.', 'R.', 'S.', 'T.', 'U.', 'V.', 'W.', 'X.', 'Y.', 'Z.']

norm_lines = []
for i, question in enumerate(dataset):

    passage = dataset[question]['passage']
    all_answers = [d[0] for d in dataset[question]['answers']]
    all_idx = [letter_index[idx] for idx, d in enumerate(dataset[question]['answers'])]
    # all_idx_b = [letter_index_b[idx] for idx, d in enumerate(dataset[question]['answers'])]
    # all_idx_c = [letter_index_c[idx] for idx, d in enumerate(dataset[question]['answers'])]

    if predictions[i] != "":
        prediction = predictions[i].lower()
    else:
        continue

    for idx, ans in zip(all_idx, all_answers):
        idx = idx.lower()
        # idx_b = idx_b.lower()
        # idx_c = idx_c.lower()
        ans = ans.lower()

        # if (idx_c in prediction) or (idx_b in prediction) or (idx in prediction) or (ans in prediction):
        if (idx in prediction) or (ans in prediction):
            norm_lines.append('yes\n')
        else:
            norm_lines.append('no\n')

with open(output_path+"/"+model_name+".norm.output.txt", "w+") as f:
    f.writelines(norm_lines)
    