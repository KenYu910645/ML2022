import pandas 

MAX_CANDI = 5 # Use top-3 candidate to relocate prediction
SUB_FILE = "cuda_1_submission.csv"
LOG_FILE = "cuda_1_submission.csv.logits"

num_preds = {} # defaultdict(int)

df_argmax = pandas.read_csv(SUB_FILE)
for c in range(10):
    print(df_argmax.loc[df_argmax['label'] == c].shape[0])
    num_preds[c] = df_argmax.loc[df_argmax['label'] == c].shape[0]


print(num_preds)
df_score = pandas.read_csv(LOG_FILE)

# Fix overflow classes
for c in range(10):
    if num_preds[c] > 10000: # Need to fix it
        print(f"Try to fix class {c}")
        candi = df_argmax.loc[df_argmax['label'] == c, 'id'].to_list()
        
        # how to pick kicked-out candidate
        candi_score = df_score.loc[df_score['id'].isin(candi)]
        candi_score = candi_score.sort_values(by=[f'c_{c}'], ascending = False)
        #
        kicked_score = candi_score.iloc[10000:]
        
        for index, row in kicked_score.iterrows():
            id = int(row['id'])
            for rank in range(1, MAX_CANDI):
                new_cls = row[1:].nlargest(n = MAX_CANDI).index[rank] # Second biggest
                if num_preds[int(new_cls[-1])] < 10000: # This is a good cls to go.
                    # Move to new cls
                    num_preds[c] -= 1
                    num_preds[int(new_cls[-1])] += 1
                    # 
                    df_argmax.loc[df_argmax['id'] == id, 'label'] = int(new_cls[-1])
                    # 
                    break
        print(num_preds)
df_argmax.to_csv("submission_balanced.csv", index=False)
