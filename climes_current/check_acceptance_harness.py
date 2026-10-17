"""Exercise the acceptance runner end to end with synthetic services and no GPU.

This is a harness check, not evidence that the real model/API/UI passed.
"""
import contextlib
from datetime import timedelta
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

PACKAGE=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
sys.argv=[str(PACKAGE/'check_final.py')]
checks=runpy.run_path(str(PACKAGE/'check_final.py'),run_name='offline_helpers')
ctx=checks['helpers']()
ctx['MODEL_NAME']='SYNTHETIC-TEST-DOUBLE'
fixtures=json.loads((PACKAGE/'replay_fixtures.json').read_text())['passages']
ctx['generate_passage_selection']=lambda question,passages: json.dumps({'selected_ids':[p['id'] for p in passages[:2]]})
ctx['embeddings']=object()

class Store:
    def similarity_search_with_relevance_scores(self,query,k):
        topic='cold' if 'cold' in query else 'flood' if 'flood' in query else 'heat'
        return [(SimpleNamespace(page_content=p['raw_text'],metadata={'source':p['source']}),0.9)
                for p in fixtures if p['topic']==topic]

ctx['Chroma']=lambda **kwargs:Store()

def get(url,params,headers,timeout):
    if url.endswith('/current'):
        payload={'response':{'current':checks['CURRENT']}}
    else:
        start=ctx['uppsala_today']()
        count=params.get('days',3)
        times=[f'{start+timedelta(days=i)}T{hour}:00' for i in range(count) for hour in ('00','12')]
        payload={'response':{'data':{'hourly':{'time':times,'temperature_2m':[2,10]*count,'precipitation':[0,1]*count}}}}
    return SimpleNamespace(status_code=200,json=lambda:payload)

ctx['requests'].get=get
# Message normalization matches the observed Gradio 6 input shape. This remains a stub.
ctx['chatbot']=SimpleNamespace(postprocess=lambda x:x,preprocess=lambda x:[{**p,'content':[] if p['content']=='' else p['content']} for p in x])
callback=next(n for n in __import__('ast').walk(__import__('ast').parse((PACKAGE/'integrated_final.py').read_text())) if isinstance(n,__import__('ast').FunctionDef) and n.name=='bot_streaming')
exec(compile(__import__('ast').Module(body=[callback],type_ignores=[]),'callback_stub','exec'),ctx)
spec=importlib.util.spec_from_file_location('acceptance_runner',PACKAGE/'START.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
real_run_path=runpy.run_path

def fake_run_path(path,run_name):
    assert Path(path)==PACKAGE/'integrated_final.py'
    return {'bot_response':ctx['bot_response'],'demo':SimpleNamespace(launch=lambda **kwargs:None)}

with tempfile.TemporaryDirectory(prefix='climes_runner_') as temporary:
    print('SYNTHETIC HARNESS CHECK: model, retrieval scores, weather provider, audio, and UI are test doubles.',flush=True)
    root=Path(temporary)
    project=root/'LLMgroup2'
    (project/'knowledge').mkdir(parents=True)
    (project/'knowledge/uppsala_all_places.json').write_text('[]')
    audio_path=root/'synthetic.mp3'
    audio_path.write_bytes(b'FAKE AUDIO: HARNESS TEST ONLY')
    ctx['generate_audio']=lambda text:str(audio_path)
    old_directory=Path.cwd()
    try:
        with patch.object(runner.runpy,'run_path',side_effect=fake_run_path):
            report,archive=runner.main(package=PACKAGE,project=project,output_root=root,open_ui=False,download=False)
        assert len(report['cases'])==14
        assert all(r['status']=='PASS' for r in report['cases']),report['cases']
        assert report['model_selections']==4
        assert report['selection_fallbacks']==0
        assert report['audio_check']['available']
        with ZipFile(archive) as bundle:
            assert bundle.testzip() is None
            assert {'acceptance.json','answers.md','events.jsonl','offline_checks.txt'}<=set(bundle.namelist())
            events=[json.loads(line) for line in bundle.read('events.jsonl').decode().splitlines()]
        assert len({r['turn_id'] for r in report['cases']})==14
        assert all(r['turn_id'] for r in report['cases'])
        for case in report['cases']:
            turn=[e for e in events if e['turn_id']==case['turn_id']]
            assert len([e for e in turn if e['event']=='question'])==1
            assert next(e['text'] for e in turn if e['event']=='text_ready')==case['answer']
        # An unexpected successful response during a simulated failure must not be scored as a pass.
        weather={'status':'unavailable','place_name':'Uppsala','intent':'current'}
        bad=[{'event':'route','route':'weather','location':{'found':True},'detail':None},
             {'event':'weather_tool','result':weather}]
        assert runner.validate_case(ctx,'weather','Invented 7.4°C',bad)[0]=='FAIL'
        print('PASS: synthetic 14-case batch, per-turn evidence, ZIP export, and false-pass rejection.')
        print('No real model, API, speech, browser, or Gradio UI was run by this harness.')
    finally:
        os.chdir(old_directory)
