import PyPDF2
import os
import json
import dashscope
import random
from http import HTTPStatus
import hanlp
from xpinyin import Pinyin
from shutil import copyfile
from time import sleep

p = Pinyin()
dashscope.api_key = 'sk-0793fab51dbd44f1a3dbf2e0541990f9'
slice_size = 9000

def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True

# 将pdf文件夹中的pdf文件转换为txt文件
def pdf_to_txt(pdf_folder, txt_folder):
    pdf_files = os.listdir(pdf_folder)
    for pdf_file in pdf_files:
        if not pdf_file.endswith('.pdf'):
            continue
        pdf_path = pdf_folder + pdf_file
        txt_path = txt_folder + pdf_file[:-4] + '.txt'
        if not os.path.exists(txt_folder):
            os.makedirs(txt_folder)
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    txt_file.write(page.extract_text())

# 进行一步简单的预处理：去除空行，去除多余空格
def format_files(txt_file, relations_file):
    with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    with open(txt_file, 'w', encoding='utf-8') as file:
        for line in lines:
            line = line.strip()
            if line:
                file.write(line + '\n')
    with open(txt_file, 'r', encoding='utf-8') as file:
        content = file.read()
    if len(content) > slice_size:
        with open(relations_file, 'r', encoding='utf-8') as file:
            relations = json.load(file)
        slices = [content[i:i+slice_size] for i in range(0, len(content), slice_size)]
        for i, slice in enumerate(slices):
            with open(txt_file[:-4] + f'_{i}.txt', 'w', encoding='utf-8') as file:
                file.write(slice)
            relations[os.path.basename(txt_file[:-4]) + f'_{i}'] = relations[os.path.basename(txt_file[:-4])]
        os.remove(txt_file)
        with open(relations_file, 'w', encoding='utf-8') as file:
            json.dump(relations, file, ensure_ascii=False)
            

# 将一个左边一列是文件名（无后缀）、右边一列是来源的csv文件转换成字典并存入relations.json
def csv_to_json(csv_file):
    relations = {}
    with open(csv_file, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip().split(',')
        relations[line[0]] = line[1]
    with open('relations.json', 'w', encoding='utf-8') as file:
        json.dump(relations, file, ensure_ascii=False)

# 单分类法尝试对data_folder文件夹中的文件进行初步分类，分类类别在categories.json中（读入成为list）
def classify_files(data_folder):
    with open('categories.json', 'r', encoding='utf-8') as file:
        categories = json.load(file)
    data_files = os.listdir(data_folder)
    for data_file in data_files:
        if not data_file.endswith('.txt'):
            continue
        with open(data_folder + data_file, 'r', encoding='utf-8') as file:
            content = file.read()
        messages = [
            {'role': 'user', 'content': f'你是一名文章分类员，请你对下面的文章进行分类：{content}\n你可以使用的类别有：{categories}，你只需要直接回答我类别即可，注意类别一定要用双引号\"\"包裹，不论一个文件同时符合一个还是多个分类，请务必在[]内返回所有分类，如：[\"{categories[0]}\"]，[\"{categories[0]}\", \"{categories[1]}\"]。如果你不知道该如何分类，可以回答[\"未知\"]'},]
        sleep(10)
        response = dashscope.Generation.call(
            'qwen1.5-32b-chat',
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='text',
        )
        try:
            json.loads(str(response))['output']['text']
        except:
            print(f'分类错误 {str(response)}')
            if not os.path.exists(data_folder + "weizhi"):
                os.mkdir(data_folder + "weizhi")
            copyfile(data_folder + data_file, data_folder + "weizhi" + '/' + data_file)
            os.remove(data_folder + data_file)
            continue
        res = json.loads(str(response))['output']['text']
        if response.status_code == HTTPStatus.OK:
            if not is_json(res):
                print(f'分类错误 {res}')
                res = ['未知']
            else:
                res = json.loads(res)
            if type(res) != list or any([x not in categories and x != '未知' for x in res]):
                print(f'分类错误 {type(res)}: {res}')
                res = ['未知']
            for category in res:
                if not os.path.exists(data_folder + p.get_pinyin(category,'')):
                    os.mkdir(data_folder + p.get_pinyin(category,''))
                copyfile(data_folder + data_file, data_folder + p.get_pinyin(category,'') + '/' + data_file)
            os.remove(data_folder + data_file)
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))

# 时间戳功能暂时不考虑实现，因为时间有限，会在后续版本中加入

# 对data_folder的所有子文件夹中的所有文件进行一次词性标注并去除多余成分以及标点符号（提前优化），之后存入datas文件夹中
def pre_process(data_folder):
    pos = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
    for filepath, dirnames, filenames in os.walk(data_folder):
        for filename in filenames:
            file = os.path.join(filepath, filename)
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            res = pos(content, tasks='pos/pku')
            for i in range(len(res['tok/fine'])):
                if res['pos/pku'][i][0] not in ['n', 'v', 'a', 'd', 'r', 'N', 'A']:
                    res['tok/fine'][i] = ''
            res = ''.join(res['tok/fine']).replace('\n', '')
            file = file.replace(data_folder, 'datas/')
            if not os.path.exists(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))
            with open(file, 'w', encoding='utf-8') as f:
                f.write(res)

if __name__ == '__main__':
    # csv_to_json('relations.csv')
    # pdf_to_txt('pdfs/', 'txts/')
    # for file in os.listdir('txts/'):
    #     if file.endswith('.txt'):
    #         format_files('txts/' + file, 'relations.json')
    # classify_files('txts/')
    pre_process('txts/')