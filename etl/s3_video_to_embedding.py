import boto3
import sys
import re
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import concat, current_timestamp, lit, udf
from pyspark.sql.types import *
from twelvelabs import TwelveLabs

# Input parameters
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'TWELVE_LABS_API_KEY', 'VIDEO_PATH', 'VIDEO_URL_PREFIX', 'EMBEDDING_PATH'])

# Generate embedding from a video
def generate_embedding(path, file_or_url='file'):
    twelvelabs_client = TwelveLabs(api_key=args['TWELVE_LABS_API_KEY'])
    params = {
        'engine_name': "Marengo-retrieval-2.6",
        'video_clip_length': 10,
    }
    params['video_file' if file_or_url=='file' else 'video_url'] = path
    task = twelvelabs_client.embed.task.create(**params)
    task.wait_for_done()
    task_result = twelvelabs_client.embed.task.retrieve(task.id)
    return [
        {
            'engine': task_result.engine_name,
            'task_status': task_result.status,
            'embedding': v.values,
            'start_offset_sec': v.start_offset_sec,
            'end_offset_sec': v.end_offset_sec,
            'embedding_scope': v.embedding_scope
        }
        for v in task_result.video_embeddings
    ] if task_result.video_embeddings else [
        {
            'engine': task_result.engine_name,
            'task_status': task_result.status,
            'embedding': None,
            'start_offset_sec': None,
            'end_offset_sec': None,
            'embedding_scope': None
        }
    ]

# UDF for generate embedding for a video    
generate_embedding_udf = udf(
    generate_embedding, 
    ArrayType(StructType([
        StructField("engine", StringType(), True),
        StructField("task_status", StringType(), True),
        StructField("embedding", ArrayType(FloatType(), True)),
        StructField("start_offset_sec", FloatType(), True),
        StructField("end_offset_sec", FloatType(), True),
        StructField("embedding_scope", StringType(), True)                 
    ])))

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)
# Fetch video list from S3
s3_client = boto3.client('s3')
paginator = s3_client.get_paginator('list_objects_v2')
match = re.match(r's3://([^/]+)/(.+)', args['VIDEO_PATH'])
video_bucket = match.group(1)
video_prefix = match.group(2)
pages = paginator.paginate(Bucket=video_bucket, Prefix=video_prefix)
videos = [
    {
        's3Key': obj['Key'],
        's3LastModifiedOn': obj['LastModified'],
        's3ETag': obj['ETag'],
        's3Size': obj['Size'],
    }
    for page in pages for obj in page.get('Contents', []) if obj['Key'].endswith('.mp4')
]
# Convert video list to a dataframe
schema = StructType(
    [
        StructField('s3Key', StringType(), False),
        StructField('s3LastModified', StringType(), False),
        StructField('s3ETag', StringType(), False),
        StructField('s3Size', IntegerType(), False),
    ]
)
df = spark.createDataFrame(videos, schema)
# Filter processed videos
match = re.match(r's3://([^/]+)/(.+)', args['EMBEDDiNG_PATH'])
if 'Contents' in s3_client.list_objects(Bucket=match.group(1), Prefix=match.group(2), Delimiter='/', MaxKeys=1):
    processed_df = spark.read.parquet(args['EMBEDDING_PATH'])
    df = df.join(processed_df, on='s3Key', how='anti')
# Prepare dataframe for embedding extraction
df = df.withColumn('processedOn', current_timestamp)
df = df.withColumn('url', concat(lit(args['VIDEO_URL_PREFIX']), df.key))
# Extract embeddings
new_video_count = df.count()
print(f"new video count: {new_video_count}")
if new_video_count > 0:
    df = df.withColumn("embedding", generate_embedding_udf(df.url, 'url'))
    df.write.mode('append').parquet(args['EMBEDDING_PATH'])
