import PyPDF2
import os
import json
import dashscope
import random
from http import HTTPStatus
import hanlp

dashscope.api_key = 'sk-0793fab51dbd44f1a3dbf2e0541990f9'

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
            with open(txt_path, 'w') as txt_file:
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    txt_file.write(page.extract_text())

# 进行一步简单的预处理：去除空行，去除多余空格
def format_files(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    with open(txt_file, 'w') as file:
        for line in lines:
            line = line.strip()
            if line:
                file.write(line + '\n')

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
        with open(data_folder + data_file, 'r') as file:
            content = file.read()
        messages = [
            {'role': 'user', 'content': f'你是一名文章分类员，请你对下面的文章进行分类：{content}\n你可以使用的类别有：{categories}，你只需要直接回答我类别即可，如：{categories[0]}。如果你不知道该如何分类，可以回答“不知道”'},]
        response = dashscope.Generation.call(
            'qwen1.5-14b-chat',
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='text',
        )
        res = json.loads(str(response))['output']['text']
        if response.status_code == HTTPStatus.OK:
            if res not in categories and res != '不知道':
                print(f'分类错误：{res}')
                continue
            if res == '不知道':
                res = '未知'
            if not os.path.exists(data_folder + res):
                os.mkdir(data_folder + res)
            os.rename(data_folder + data_file, data_folder + res + '/' + data_file)
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
            with open(file, 'r') as f:
                content = f.read()
            res = pos(content, tasks='pos/pku')
            for i in range(len(res['tok/fine'])):
                if res['pos/pku'][i][0] not in ['n', 'v', 'a', 'd', 'r', 'N', 'A']:
                    res['tok/fine'][i] = ''
            res = ''.join(res['tok/fine']).replace('\n', '')
            file = file.replace(data_folder, 'datas/')
            if not os.path.exists(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))
            with open(file, 'w') as f:
                f.write(res)

if __name__ == '__main__':
    pdf_to_txt('pdfs/', 'txts/')
    for file in os.listdir('txts/'):
        if file.endswith('.txt'):
            format_files('txts/' + file)
    csv_to_json('relations.csv')
    classify_files('txts/')
    pre_process('txts/')