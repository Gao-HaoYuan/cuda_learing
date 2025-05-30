import os
import argparse

def delete_files(src, file_type):
    """
    遍历源文件夹，并删除指定类型的源文件。

    :param src: 源目录路径
    :param file_type: 文件类型（后缀名）
    """
    filelist = os.listdir(src)

    for file in filelist:
        file_name = os.path.join(src, file)
        
        # 如果是目录，递归调用
        if os.path.isdir(file_name):
            delete_files(file_name, file_type)
            # 防止目录名包含 file_type
            continue

        if file.endswith(file_type):
            os.remove(file_name)
            print(f"Deleted: {file_name}")

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Delete files of a specific type in a directory.")
    
    # 添加参数
    parser.add_argument('src', type=str, help="The source directory path.")
    parser.add_argument('file_type', type=str, help="The file type (e.g., '.c').")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用删除文件的函数
    delete_files(args.src, args.file_type)

if __name__ == "__main__":
    main()