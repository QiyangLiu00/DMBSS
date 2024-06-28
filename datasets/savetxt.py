import os


def generate(dir, label):
    files = os.listdir(dir)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    files.sort()  # 对文件或文件夹进行排序
    print('****************')
    print('input :', dir)
    print('start...')
    listText = open('../Data/PatternNet/PatternNet.txt', 'a+')  # 创建并打开一个txt文件，a+表示打开一个文件并追加内容
    testText = open('../Data/PatternNet/PatternNet_test_list.txt', 'a+')
    trainText = open('../Data/PatternNet/PatternNet_train_list.txt', 'a+')
    num = len(files)
    r = 0.2
    testnum = num * r
    j = 0
    for file in files:  # 遍历文件夹中的文件
        fileType = os.path.split(file)  # os.path.split（）返回文件的路径和文件名，【0】为路径，【1】为文件名
        if fileType[1] == '.txt':  # 若文件名的后缀为txt,则继续遍历循环，否则退出循环
            continue
        name = dir + '/' + file + ' ' + str(int(label)) + '\n'  # name 为文件路径和文件名+空格+label+换行
        if j < testnum:
            testText.write(name)
        else:
            trainText.write(name)
        j += 1
        listText.write(name)  # 在创建的txt文件中写入name
    listText.close()  # 关闭txt文件
    # testText.close()
    # trainText.close()
    print('down!')
    print('****************')


if __name__ == '__main__':  # 主函数
    # 数据集路径：
    # F:/LQY/NWPU-RESISC45
    outer_path = 'F:/LQY/PatternNet'  # 这里数据集路径
    i = 0
    listText = open('../Data/PatternNet/class.txt', 'a+')  # 创建并打开一个txt文件，a+表示打开一个文件并追加内容
    folderlist = os.listdir(outer_path)  # 列举文件夹
    for folder in folderlist:  # 遍历文件夹中的文件夹(若engagement文件夹中存在txt或py文件，则后面会报错）
        generate(os.path.join(outer_path, folder), i)  # 调用generate函数，函数中的参数为：（图片路径+文件夹名，标签号）
        name = folder + ' ' + str(int(i)) + '\n'  # name 为文件路径和文件名+空格+label+换行
        i += 1
        listText.write(name)  # 在创建的txt文件中写入name
    listText.close()  # 关闭txt文件
    print('down!')
