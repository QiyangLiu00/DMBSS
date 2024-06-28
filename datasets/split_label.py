# 取出标签名，并保留前5个字母
def keep_first_four_letters(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 只保留每行的前4个字母
    new_lines = [line[:5].upper() + '\n' for line in lines]

    # 将结果写入新的txt文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)


if __name__ == "__main__":
    # 用法示例
    input_file = '../Data/PatternNet/class.txt'  # 输入文件名
    output_file = '../Data/PatternNet/label.txt'  # 输出文件名
    keep_first_four_letters(input_file, output_file)
