from typing import List

# Z-Score Normalization


def normalize_data(data): return (data - data.mean()) / data.std()

# Max Min Normalization (Giving worse results)
# def normalize_data(data: pandas.DataFrame):
#     min_val = data.min()
#     max_val = data.max()

#     normalized_data = (data - min_val) / (max_val - min_val)
#     return normalized_data


def big_print(string: str, data: any):
    print(f"<===================={string}====================>\n")
    print(data)
    print("\n-----------------------------------------------------\n")


def max_print(my_list):
    max_value = max(my_list)
    print(
        f"Maximum accuracy: {100 * round(max_value, 5)} at epoch {1 + my_list.index(max_value)}")
    print(f"Final Accuracy: {100 * round(my_list[-1], 5)}")
