import os
import json

def combine_json_dicts(folder_path, output_file):
    combined_dict = {}

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 查找所有名为 output_dict.json 的文件
            if file == "output_dict.json":
                file_path = os.path.join(root, file)
                
                # 打开并读取 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        new_datas={}
                        template=file_path.split('/')[-2]+'.'
                        new_datas[template]=[]
                        datas = json.load(f)
                        for data in datas[template]:
                            data['question']=data['question'].replace('[X]',data['sub_label'])
                            data['answer']=data['obj_label']
                            data.pop('sub_label')
                            data.pop('obj_label')
                            new_datas[template].append(data)
                        
                        
                        # 将当前字典数据合并到 combined_dict 中
                        combined_dict.update(new_datas)
                    except json.JSONDecodeError as e:
                        print(f"无法解析文件 {file_path}: {e}")

    # 将最终合并的字典输出到一个新文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_dict, f, ensure_ascii=False, indent=1)

    print(f"所有数据已成功合并到 {output_file}")

# 使用示例
folder_path = "data/lama_dicts"  # 替换为你的文件夹路径
output_file = "data/lama/counter-lama.json"  # 输出文件名
combine_json_dicts(folder_path, output_file)
