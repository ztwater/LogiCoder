import os

from globals import Globals
from myutils import FileUtils, DataUtils


def extract_completion(N):
    """
    Extract LLMs' completion results from output jsonline.
    """
    codereval_metadata = FileUtils.read_jsonl(Globals.CODEREVAL_BENCHMARK_METADATA_PATH)
    for file in os.listdir(Globals.PREDICTION_DIR):
        if file.endswith(f'N{N}.jsonl'):
            # print(file)
            file_path = os.path.join(Globals.PREDICTION_DIR, file)
            lines = FileUtils.read_jsonl(file_path)
            new_lines = []
            for line in lines:
                namespace = line['metadata']['namespace']
                func_name = namespace.split('.')[-1]
                indent = line['metadata']['indent']
                generated_results = []
                for i, pred in enumerate(line['predictions']):
                    output = pred['text']
                    # if output is None:
                    #     print(pred)
                    func = DataUtils.extract_function_from_output(output, func_name)
                    completion = DataUtils.extract_completion(output if func == "" and output is not None else func, indent)

                    if 'codereval' in file_path:
                        generated_results.append(func)
                    else:
                        new_lines.append({
                            'namespace': namespace,
                            'completion': completion,
                            'idx': i
                        })
                if 'codereval' in file_path:
                    _id = namespace.split('.')[0]
                    for md in codereval_metadata:
                        if md['namespace'] == namespace:
                            _id = md['_id']
                    new_lines.append({
                        '_id': _id,
                        'generate_results': generated_results
                    })
            file_name = '.'.join(file.split('.')[:-1])
            completion_path = os.path.join(Globals.PREDICTION_DIR, file_name, f'{file_name}.jsonl')
            FileUtils.dump_jsonl(new_lines, completion_path)


if __name__ == '__main__':
    extract_completion(1)

    # tasks = FileUtils.read_jsonl(Globals.OUR_BENCHMARK_METADATA_PATH)
    # # tasks = FileUtils.read_jsonl(Globals.FULL_BENCHMARK_METADATA_PATH)
    # # # tasks = FileUtils.read_jsonl(Globals.CODEREVAL_BENCHMARK_METADATA_PATH)
    # repo_base_dir = Globals.REPO_BASE_DIR # Globals.REPO_BASE_CODEREVAL_DIR
    # #
    # group tasks by repos
    # repo_to_task_mappings = dict()
    # for t in tasks:
    #     repo = t['repo_path']
    #     repo_to_task_mappings.setdefault(repo, [])
    #     repo_to_task_mappings[repo].append(t)

    # repo_list = []
    # for repo_path, task_list in repo_to_task_mappings.items():
    #     tokens = FileUtils.find_py_files(os.path.join(repo_base_dir, repo_path))
    #     repo_list.append([repo_path, tokens])
    # repo_list = sorted(repo_list, key=lambda x:x[1])
    # print(repo_list)
    #     # if tokens > 1024*1024:
    #     #     repos['>1M'].append(repo_path)
    #     # elif tokens > 256*1024:
    #     #     repos['256K-1M'].append(repo_path)
    #     # elif tokens > 256*1024:
    #     #     repos['128K-256K'].append(repo_path)
    #     # else:
    #     #     repos['<128K'].append(repo_path)
    # # print(repos)
    #
    # codereval_dict = {'>1M': ['openstack---cinder', 'santoshphilip---eppy', 'rougier---matplotlib'], '256K-1M': ['neo4j---neo4j-python-driver', 'rak-n-rok---Krake', 'scieloorg---packtools', 'mozilla---relman-auto-nag', 'openstack---neutron-lib', 'mwatts15---rdflib', 'awsteiner---o2sclpy'], '128K-256K': [], '<128K': ['pre-commit---pre-commit', 'pexip---os-python-cachetools', 'champax---pysolbase', 'pexip---os-zope', 'ynikitenko---lena', 'cpburnz---python-sql-parameters', 'zimeon---ocfl-py', 'bastikr---boolean', 'pexip---os-python-dateutil', 'eykd---prestoplot', 'witten---atticmatic', 'witten---borgmatic', 'infobloxopen---infoblox-client', 'SoftwareHeritage---swh-lister', 'ossobv---planb', 'burgerbecky---makeprojects', 'skorokithakis---shortuuid', 'sipwise---repoapi', 'turicas---rows', 'cloudmesh---cloudmesh-common', 'ikus060---rdiffweb', 'commandline---flashbake', 'bazaar-projects---docopt-ng', 'kirankotari---shconfparser', 'ansible-security---ansible_collections.ibm.qradar', 'scrolltech---apphelpers', 'ufo-kit---concert', 'MozillaSecurity---lithium', 'jaywink---federation', 'redhat-openstack---infrared']}
    # deveval_dict = {'>1M': ['Internet/boto', 'Software-Development/Faker', 'Software-Development/Django', 'Scientific-Engineering/wandb', 'Communications/twilio-fatisar', 'Scientific-Engineering/cupy', 'Scientific-Engineering/diffusers'], '256K-1M': ['Internet/falcon', 'Internet/Authlib', 'Internet/djangorestframework', 'Internet/pyramid', 'Internet/kinto', 'Database/alembic', 'Database/arctic-latest', 'System/mrjob', 'Security/zxcvbn-python', 'Security/capirca', 'Security/asyncssh', 'Security/msticpy', 'Software-Development/discord-py', 'Software-Development/dash', 'Software-Development/peewee', 'Scientific-Engineering/bentoml', 'Scientific-Engineering/pymc', 'Utilities/jc', 'Utilities/mmcv', 'Communications/Telethon', 'Communications/aioxmpp', 'Scientific-Engineering/datasets', 'Database/mongoengine', 'Security/barf', 'Security/oletools', 'Software-Development/backtrader'], '128K-256K': [], '<128K': ['Text-Processing/python-benedict', 'Text-Processing/feedparser', 'Text-Processing/mistune', 'Text-Processing/xmnlp', 'Text-Processing/parsel', 'Text-Processing/dominate', 'Text-Processing/rows', 'Text-Processing/pycorrector', 'Text-Processing/natasha', 'Internet/google-api-python-client', 'Internet/Jinja2', 'Internet/sumy', 'Internet/djangorestframework-simplejwt', 'Internet/proxybroker', 'Database/csvs-to-sqlite', 'Database/sqlitedict', 'Database/litecli', 'Database/happybase', 'Database/mssql-cli', 'Database/datasette', 'Database/mongo-doc-manager', 'Database/bplustree', 'Multimedia/psd-tools', 'Database/sqlite-utils', 'Multimedia/Mopidy', 'Multimedia/hypertools', 'Multimedia/gif-for-cli', 'Multimedia/mingus', 'System/exodus-bundler', 'System/fs', 'System/wal-e', 'System/pyinfra', 'System/sshuttle', 'System/flower', 'Security/trailscraper', 'Security/pycoin', 'Security/python-taint', 'Security/diffprivlib', 'System/sslyze', 'Software-Development/ydata-profiling', 'Software-Development/PySnooper', 'Software-Development/albumentations', 'Scientific-Engineering/csvkit', 'Scientific-Engineering/folium', 'Scientific-Engineering/TPOT', 'Utilities/PyJWT', 'Utilities/pytube', 'Utilities/sacred', 'Utilities/boltons', 'Utilities/gunicorn', 'Utilities/praw', 'Utilities/python-for-android', 'Utilities/mackup', 'Utilities/stellar', 'Communications/IMAPClient', 'Communications/hbmqtt', 'Communications/zulip-term', 'Communications/chatette', 'System/prometheus-client', 'Text-Processing/pymorphy2', 'Text-Processing/PyLaTeX', 'System/trackerjacker', 'Database/awesome-autodl', 'Communications/Wikipedia-API', 'Communications/twtxt', 'Scientific-Engineering/rq', 'Text-Processing/online-judge-tools', 'Communications/PySimpleSOAP', 'Security/principalmapper', 'Software-Development/pandas-profiling', 'Scientific-Engineering/lux', 'System/viztracer', 'Security/threatingestor', 'Communications/ehforwarderbot', 'Communications/hl7', 'Utilities/whereami', 'Utilities/pymusic-dl', 'Security/passpie', 'Internet/python-twitter', 'Security/pyOpenSSL', 'Database/asyncpg', 'Internet/databases']}
    # for k, v in codereval_dict.items():
    #     print(k, len(v))
    # for k, v in deveval_dict.items():
    #     print(k, len(v))