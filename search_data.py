from pymilvus import MilvusClient
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from xpinyin import Pinyin
import dashscope
from http import HTTPStatus
import random
import os

collection_name = "test_collection"
EMBEDDING_DEVICE = "cuda"
dashscope.api_key = 'sk-0793fab51dbd44f1a3dbf2e0541990f9'
p = Pinyin()
limit = 2

client = MilvusClient(
    uri="http://114.212.97.40:19530",
)

def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True

def partition_search(query):
    with open('categories.json', 'r', encoding='utf-8') as file:
        categories = json.load(file)
    messages = [
        {'role': 'user', 'content': f'你是一名问题分类员，请你对下面的问题进行分类：{query}\n你可以使用的类别有：{categories}，你只需要直接回答我类别即可，注意类别一定要用双引号\"\"包裹，不论一个文件同时符合一个还是多个分类，请务必在[]内返回所有分类，如：[\"{categories[0]}\"]，[\"{categories[0]}\", \"{categories[1]}\"]。如果你不知道该如何分类，可以回答[\"未知\"]'},]
    response = dashscope.Generation.call(
        'qwen1.5-32b-chat',
        messages=messages,
        seed=random.randint(1, 10000),
        result_format='text',
    )
    res = json.loads(str(response))['output']['text']
    if response.status_code == HTTPStatus.OK:
        if not is_json(res):
            return general_search(query)
        else:
            res = json.loads(res)
        if type(res) != list or any([x not in categories for x in res]):
            return general_search(query)
        embeddings = HuggingFaceEmbeddings(model_name="./m3e-base", model_kwargs={'device': EMBEDDING_DEVICE})
        embedding = embeddings.embed_documents([query])
        results = client.search(collection_name=collection_name, data=embedding, limit=limit, search_params={"metric_type": "COSINE", "params": {}}, output_fields=["file_path"], partition_names=[p.get_pinyin(category,'') for category in res])
        return [result['entity']['file_path'].replace('datas','txts') for result in results[0]]
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

def general_search(query):
    embeddings = HuggingFaceEmbeddings(model_name="./m3e-base", model_kwargs={'device': EMBEDDING_DEVICE})
    embedding = embeddings.embed_documents([query])
    results = client.search(collection_name=collection_name, data=embedding, limit=limit, search_params={"metric_type": "COSINE", "params": {}}, output_fields=["file_path"])
    return [result['entity']['file_path'].replace('datas','txts') for result in results[0]]

def final_work(files, query):
    files_content = ""
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            files_content += f.read()
    messages = [
        {'role': 'user', 'content': f'{files_content}\n请你依据以上所有内容回答这个问题：{query}\n如果以上内容无法确定答案，请回答"对不起，我不知道这个问题的答案。"，不要回答未提供的内容。'},]
    responses = dashscope.Generation.call(
        'qwen1.5-32b-chat',
        messages=messages,
        seed=random.randint(1, 10000),
        result_format='text',
        stream=True,
        output_in_full=False,
    )
    res = ''
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            print(json.loads(str(response))['output']['text'], end='')
            res += json.loads(str(response))['output']['text']
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    if res.find('对不起，我不知道这个问题的答案。') != -1:
        return ''
    res = '\nSOURCE: '
    with open('relations.json', 'r', encoding='utf-8') as file:
        relations = json.load(file)
    for file in files:
        fname = os.path.basename(file).split('.')[0]
        if fname in relations.keys():
            res += f'[{fname}]({relations[fname]})\n'
    return res

if __name__ == '__main__':
    query = input("请输入问题：")
    print(final_work(partition_search(query), query))