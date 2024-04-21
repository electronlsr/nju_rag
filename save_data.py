from pymilvus import MilvusClient, DataType, FieldSchema
import json
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from xpinyin import Pinyin

EMBEDDING_DEVICE = "cuda"
chunk_size = 30
p = Pinyin()

# 连接 Milvus
client = MilvusClient(
    uri="https://in01-a6f9ecd045273fa.gcp-asia-southeast1.vectordb.zillizcloud.com:443",
    token="db_admin:Gb3(}kK(&CoPN6hm"
)

# 创建 collection，并用collection_name.json中的分类作为分区名创建分区
def create_collection(collection_name):
    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768*chunk_size)
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
        params={"nlist": 128}
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    with open('categories.json', 'r', encoding='utf-8') as file:
        categories = json.load(file)
    for category in categories:
        client.create_partition(collection_name, p.get_pinyin(category,''))
    print(client.list_partitions(collection_name))

# 将datas文件夹中的所有文件的内容进行embedding并存入collection_name中的对应分区
def insert_data(collection_name):
    embeddings = HuggingFaceEmbeddings(model_name="./m3e-base", model_kwargs={'device': EMBEDDING_DEVICE})
    for filepath, dirnames, filenames in os.walk('datas'):
        for filename in filenames:
            with open(filepath + '/' + filename, 'r') as file:
                content = file.read()
            partition_name = p.get_pinyin(os.path.basename(filepath),'')
            # 将content切片成长度为chunk_size的分块并依次插入
            chunks = [content[i:i+chunk_size] for i in range(0, len(content) - chunk_size, chunk_size)]
            for chunk in chunks:
                embedding = embeddings.embed_documents(chunk)
                embedding = [item for sublist in embedding for item in sublist]
                client.insert(collection_name=collection_name, data=[{"embedding": embedding, "file_path": filepath}], partition_name=partition_name)
            # 补齐最后不足chunk_size的字符
            remaining_chars = len(content) % chunk_size
            last_item = content[-remaining_chars:]
            last_item = last_item * (chunk_size // remaining_chars)
            last_item += ' ' * (chunk_size - len(last_item))
            embedding = embeddings.embed_documents(last_item)
            embedding = [item for sublist in embedding for item in sublist]
            client.insert(collection_name=collection_name, data=[{"embedding": embedding, "file_path": filepath}], partition_name=partition_name)

if __name__ == '__main__':
    collection_name = 'test_collection'
    # create_collection(collection_name)
    insert_data(collection_name)