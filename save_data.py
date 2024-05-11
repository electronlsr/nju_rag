from pymilvus import MilvusClient, DataType, FieldSchema
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from xpinyin import Pinyin

EMBEDDING_DEVICE = "cuda"
chunk_size = 100
nlist = 128
p = Pinyin()
id_begin = 0

# 连接 Milvus
client = MilvusClient(
    uri="http://114.212.97.40:19530",
)

# 创建 collection，并用categories.json中的分类作为分区名创建分区
def create_collection(collection_name):
    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
    schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="id",
        index_type="STL_SORT"
    )
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": nlist}
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    with open('categories.json', 'r', encoding='utf-8') as file:
        categories = json.load(file)
    for category in categories:
        client.create_partition(collection_name, p.get_pinyin(category,''))
    print(client.list_partitions(collection_name))

# 删除 collection
def drop_collection(collection_name):
    client.drop_collection(collection_name)

# 将datas文件夹中的所有文件的内容进行embedding并存入collection_name中的对应分区，将返回的对应id存入id_list.json中
def insert_data(collection_name):
    embeddings = HuggingFaceEmbeddings(model_name="./m3e-base", model_kwargs={'device': EMBEDDING_DEVICE})
    idl = []
    global id_begin
    for filepath, dirnames, filenames in os.walk('datas'):
        for filename in filenames:
            with open(filepath + '/' + filename, 'r', encoding='utf-8') as file:
                content = file.read()
            data = []
            partition_name = p.get_pinyin(os.path.basename(filepath),'')
            # 将content切片成长度为chunk_size的分块并依次插入
            chunks = [content[i:i+chunk_size] for i in range(0, len(content) - chunk_size, chunk_size)]
            for chunk in chunks:
                embedding = embeddings.embed_documents([chunk])[0]
                data.append({"embedding": embedding, "file_path": filepath + '/' + filename})
            # 补齐最后不足chunk_size的字符
            remaining_chars = len(content) % chunk_size
            last_item = content[-remaining_chars:]
            embedding = embeddings.embed_documents([last_item])[0]
            data.append({"embedding": embedding, "file_path": filepath + '/' + filename})
            idl.append({'id': id_begin, 'file_path': filepath + '/' + filename,'ids': list(client.insert(collection_name=collection_name, data=data, partition_name=partition_name)['ids'])})
            id_begin += 1
    with open('id_list.json', 'w', encoding='utf-8') as file:
        json.dump(idl, file, ensure_ascii=False)

# 删除id_list.json中的对应id，并删除collection_name中的对应数据
def delete_data(collection_name, data_id):
    with open('id_list.json', 'r', encoding='utf-8') as file:
        idl = json.load(file)
    for item in idl:
        if item['id'] == data_id:
            print(client.delete(collection_name=collection_name, ids=item['ids']))
            idl.remove(item)
            break
    with open('id_list.json', 'w', encoding='utf-8') as file:
        json.dump(idl, file, ensure_ascii=False)

# 删除collection_name中的所有数据
def monitor_collection(collection_name):
    print(client.describe_collection(collection_name = collection_name))

# 列出所有collection
def list_collections():
    print(client.list_collections())

# 对所有collection进行monitor_collection
def monitor():
    list_collections()
    lst = client.list_collections()
    for collection in lst:
        monitor_collection(collection)

if __name__ == '__main__':
    collection_name = 'test_collection'
    create_collection(collection_name)
    insert_data(collection_name)
    # drop_collection(collection_name)
    monitor()