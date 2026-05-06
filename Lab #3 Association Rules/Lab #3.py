#Names: Trevor Henderson, Jose Cadenas

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#Dataset Link: https://www.kaggle.com/datasets/steve1215rogg/e-commerce-dataset

def items_to_text(s):
    return ", ".join(sorted(list(s)))

def support_of_itemset(onehot_df, items):
    if len(items) == 0:
        return 0.0
    return onehot_df[list(items)].all(axis=1).mean()

def main():
    # CSV must be in Downloads folder
    csv_path = "/Users/josie/Downloads/ecommerce_dataset_updated.csv"

    min_support = 0.05
    min_confidence = 0.40

    # ===== LOAD DATA =====
    df = pd.read_csv(csv_path)

    # ===== CLEAN + FEATURE ENGINEERING =====
    df = df.dropna(subset=["User_ID", "Purchase_Date", "Category", "Discount (%)"])
    df["User_ID"] = df["User_ID"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Purchase_Date"])

    df["Discount (%)"] = pd.to_numeric(df["Discount (%)"], errors="coerce").fillna(0)
    df["Has_Discount"] = df["Discount (%)"].apply(
        lambda x: "HasDiscount" if x > 0 else "NoDiscount"
    )

    # ===== BUILD TRANSACTIONS =====
    df["TransactionID"] = df["User_ID"] + "_" + df["Purchase_Date"].dt.strftime("%Y-%m-%d")

    cat_items = df[["TransactionID", "Category"]].copy()
    cat_items["Item"] = "CAT=" + cat_items["Category"]
    cat_items = cat_items[["TransactionID", "Item"]]

    disc_items = df[["TransactionID", "Has_Discount"]].copy()
    disc_items.columns = ["TransactionID", "Item"]

    df_items = pd.concat([cat_items, disc_items], ignore_index=True)

    transactions = (
        df_items.groupby("TransactionID")["Item"]
        .apply(lambda x: list(set(x)))
        .tolist()
    )

    # ===== ONE-HOT ENCODING =====
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(te_array, columns=te.columns_)

    # ===== APRIORI =====
    frequent_itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence
    )
    rules = rules.sort_values(["lift", "confidence"], ascending=False)

    # ===== OUTPUT =====
    print("DATASET")
    print("Transactions:", len(transactions))
    print("Unique items:", onehot.shape[1])
    print()

    print("FREQUENT ITEMSETS (top 10)")
    fp = frequent_itemsets.copy()
    fp["itemsets"] = fp["itemsets"].apply(items_to_text)
    fp["support"] = fp["support"].round(4)
    print(fp[["support", "itemsets"]].head(10).to_string(index=False))
    print()

    print("ASSOCIATION RULES (top 10)")
    rp = rules[["antecedents", "consequents", "support", "confidence", "lift", "conviction"]].copy()
    rp["antecedents"] = rp["antecedents"].apply(items_to_text)
    rp["consequents"] = rp["consequents"].apply(items_to_text)
    rp = rp.round(4)
    print(rp.head(10).to_string(index=False))
    print()

    # ===== MANUAL CHECK =====
    example = rules.iloc[0]
    A = example["antecedents"]
    B = example["consequents"]

    support_A = support_of_itemset(onehot, A)
    support_B = support_of_itemset(onehot, B)
    support_AB = support_of_itemset(onehot, set(A) | set(B))
    confidence = support_AB / support_A
    lift = confidence / support_B
    conviction = (1 - support_B) / (1 - confidence)

    print("EXAMPLE RULE CHECK")
    print("A:", items_to_text(A))
    print("B:", items_to_text(B))
    print("support(A):", round(support_A, 4))
    print("support(B):", round(support_B, 4))
    print("confidence:", round(confidence, 4))
    print("lift:", round(lift, 4))
    print("conviction:", round(conviction, 4))

if __name__ == "__main__":
    main()


#LAB 3 EXPLANATION:

#For this lab, we used an e-commerce dataset and applied association rule mining to
#identify patterns in customer purchases. We defined each transaction as one user's
#purchases made on the same day and used product categories as the main items.
#As part of feature engineering, we created a discount indicator to show whether an
#item was discounted. We selected minimum support and confidence values to filter
#out rare and weak patterns. The results show which categories commonly appear
#together and how discounts relate to purchasing behavior, which can be useful for
#marketing and product placement decisions.