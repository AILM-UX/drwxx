import os
import openpyxl

# 设置文件夹路径
root_dir = 'G:/dataset/total_dataset'  # 替换为你文件夹的路径

# 创建一个新的 Excel 工作簿
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Folder Names"

# 添加表头
ws.append(["Parent Folder", "Sub Folder", "Sub-Sub Folder"])

# 遍历根目录中的所有子文件夹（0到19）
for i in range(20):
    parent_folder = str(i)  # 对应文件夹名（0到19）
    parent_folder_path = os.path.join(root_dir, parent_folder, 'video_frame')

    if os.path.exists(parent_folder_path):
        # 遍历 'video_frame' 中的所有小文件夹
        for subfolder_name in os.listdir(parent_folder_path):
            subfolder_path = os.path.join(parent_folder_path, subfolder_name)

            if os.path.isdir(subfolder_path):
                # 将父文件夹、video_frame子文件夹和小文件夹的名字写入到 Excel
                ws.append([parent_folder, subfolder_name])

# 保存 Excel 文件
output_file = 'folder_names.xlsx'  # 你想保存的文件名
wb.save(output_file)

print(f"所有文件夹名字已保存到 {output_file}")