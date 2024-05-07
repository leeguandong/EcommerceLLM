import os
import json
from tqdm import tqdm


def merge_json_files(folder_path, output_file):
    # 初始化一个空列表，用于存储所有的字典
    merged_data = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 确保文件是JSON文件
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            # 读取JSON文件并将其中的字典添加到列表中
            try:
                with open(file_path, 'r', encoding="utf-8") as file:
                    json_data = json.load(file)
                    merged_data.extend(json_data)
            except:
                print("error", filename)
                continue

    # 将包含所有字典的列表保存到输出文件中
    with open(output_file, 'w', encoding="utf-8") as output:
        json.dump(merged_data, output, indent=4)


# 指定文件夹路径和输出文件路径
folder_path = 'E:/comprehensive_library/e_commerce_llm/data/tiktok_v1'
output_file = 'E:/comprehensive_library/e_commerce_llm/data/tiktok_v1.json'

# 调用函数进行合并
merge_json_files(folder_path, output_file)
