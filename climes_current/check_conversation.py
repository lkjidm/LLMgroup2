"""Reported conversation regressions; no live model, API, speech or UI."""
from pathlib import Path
import runpy, sys, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
old=sys.argv[:];sys.argv=[str(ROOT/'climes_current/check_final.py')]
legacy=runpy.run_path(sys.argv[0],run_name='conversation_helpers');sys.argv=old
import heatwave_stats as stats

class ConversationChecks(unittest.TestCase):
    def setUp(self): self.ns=legacy['helpers']()
    def test_healthcare_paraphrases(self):
        for q in ['What should the healthcare system prepare for during heatwaves?', 'What should the healthcare system prepare for during a heatwave?', 'How can hospitals prepare for extreme heat?']:
            route,d=self.ns['classify_request'](q)
            self.assertEqual(route,'advice');self.assertTrue(d['healthcare']);self.assertEqual(d['topic'],'heat')
    def test_keep_cool_no_weather(self):
        self.ns['retrieve_manual_passages']=Mock(return_value=[])
        self.ns['build_answer']('How can I keep cool during hot weather in Uppsala?')
        self.ns['retrieve_manual_passages'].assert_called_once_with('heat');self.ns['requests'].get.assert_not_called()
    def test_missing_unit_clarifies(self):
        a=self.ns['build_answer']('Can we have 3 days of 31 degrees in July?')
        self.assertIn('°C or °F',a);self.assertIn('historical',a);self.assertNotIn('unrelated requests',a)
        self.ns['requests'].get.assert_not_called()
    def test_can_we_have_does_not_guess_intent(self):
        a=self.ns['build_answer']('Can we have 3 days of 31°C in July?')
        self.assertIn('historical',a);self.assertIn('future forecast',a);self.assertNotIn('unrelated requests',a)
        self.ns['requests'].get.assert_not_called()
    def test_next_july_not_a_fabricated_forecast(self):
        a=self.ns['build_answer']('Will Uppsala have four consecutive days above 31°C next July?')
        self.assertIn('cannot confirm',a);self.assertIn('not a forecast or a probability',a)
        self.assertNotIn('unrelated requests',a);self.ns['requests'].get.assert_not_called()
    def test_month_passed_to_tool_contract(self):
        route,d=self.ns['classify_request']('Historically how often has Uppsala had 31°C for 3 consecutive days in July?')
        self.assertEqual(route,'historical');self.assertEqual(d['month'],7);self.assertEqual(d['duration_days'],3)
    def test_word_duration_and_strict_comparison(self):
        _,d=self.ns['classify_request']('Historically how often did Uppsala have above 31°C for three consecutive days in July?')
        self.assertEqual(d['duration_days'],3);self.assertEqual(d['comparison'],'above');self.assertEqual(d['month'],7)
    def test_unsupported_date_restrictions_not_ignored(self):
        for suffix in ['in July 2018','in summer','in July and August']:
            a=self.ns['build_answer']('Historically how often did Uppsala have 31°C for 3 days '+suffix+'?')
            self.assertIn('does not currently support',a);self.ns['requests'].get.assert_not_called()
    def test_cross_month_days_excluded(self):
        d=pd.DataFrame({'date':pd.date_range('2020-07-30',periods=4),'tmax':[32]*4})
        self.assertEqual(len(stats.find_heatwaves(d,31,3)),1)
        self.assertEqual(len(stats.find_heatwaves(d,31,3,month=7)),0)
        self.assertEqual(len(stats.find_heatwaves(d,31,2,month=7)),1)
    def test_different_years_never_join(self):
        d=pd.DataFrame({'date':['2019-07-30','2019-07-31','2020-07-01'],'tmax':[32]*3})
        self.assertEqual(len(stats.find_heatwaves(d,31,3,month=7)),0)
    def test_above_is_not_at_least(self):
        d=pd.DataFrame({'date':pd.date_range('2020-07-01',periods=3),'tmax':[31]*3})
        self.assertEqual(len(stats.find_heatwaves(d,31,3)),1)
        self.assertEqual(len(stats.find_heatwaves(d,31,3,comparison='above')),0)
    def test_invalid_scope(self):
        for m in [True,0,13,7.5,'July']: self.assertIn('error',stats.historical_heatwave_stats(3,31,month=m))
        self.assertIn('error',stats.historical_heatwave_stats(3,31,comparison='unknown'))
    def test_july_independent_scalar_count(self):
        d=pd.read_csv(ROOT/'uppsala_tmax_1985_2026.csv');count=length=0;previous=None
        for row in d.itertuples():
            date=pd.Timestamp(row.date);hot=date.month==7 and pd.notna(row.tmax) and row.tmax>=31
            if not hot or previous is None or (date-previous).days!=1:
                count+=length>=3;length=0
            if hot: length+=1
            previous=date
        count+=length>=3
        self.assertEqual(count,5)
        self.assertEqual(stats.historical_heatwave_stats(3,31,month=7)['historical_count'],count)
        self.assertEqual(stats.historical_heatwave_stats(3,31)['historical_count'],6)
    def test_july_display_states_scope(self):
        a=self.ns['build_answer']('Historically how often has Uppsala had 31°C for 3 consecutive days in July?')
        self.assertIn('five separate runs',a);self.assertIn('all counted days within July',a)
        self.assertNotIn('six separate runs',a);self.ns['requests'].get.assert_not_called()
    def test_known_counts_unchanged(self):
        for days,temp,count in [(4,31,2),(3,32,3),(4,35,0)]:
            self.assertEqual(stats.historical_heatwave_stats(days,temp)['historical_count'],count)
    def test_no_invented_duration(self):
        self.ns['retrieve_manual_passages']=Mock(return_value=[])
        self.ns['get_historical_context']=Mock(side_effect=AssertionError('No duration'))
        self.ns['build_answer']('Suppose it is 32°C in Uppsala. What precautions should I take?')
        self.ns['get_historical_context'].assert_not_called();self.ns['requests'].get.assert_not_called()
    def test_policy_antecedent_preserved(self):
        raw='Surge response also relies on secure access to medicines, protective equipment and essential medical consumables. This requires resilient supply chains and strategic reserves able to function during disruptions.'
        self.assertEqual(self.ns['complete_policy_sentences'](raw),[raw])
        self.assertEqual(self.ns['complete_policy_sentences']('This requires resilient supply chains and strategic reserves able to function during disruptions.'),[])
    def test_hazard_context_outranks_generic_policy(self):
        generic='A resilient health system requires adequate staffing, secure essential supplies and strategic reserves to maintain services during prolonged climate-related crises.'
        specific='Hospitals must prepare for heatwaves by maintaining service continuity and essential supplies during prolonged periods of extreme heat.'
        docs=[(SimpleNamespace(page_content=t,metadata={'source':'CLIMES_Healthcare_preparedness.pdf','page':4}),score) for t,score in [(generic,.99),(specific,.6)]]
        store=Mock();store.similarity_search_with_relevance_scores.return_value=docs
        self.ns['Chroma']=Mock(return_value=store);self.ns['embeddings']=object()
        with patch.object(self.ns['os'].path,'exists',return_value=True): result=self.ns['retrieve_manual_passages']('heat',top_k=1,healthcare=True)
        self.assertEqual(result[0]['raw_text'],specific)
    def test_healthcare_scope_disclosed(self):
        p={'source':'CLIMES_Healthcare_preparedness.pdf','page':4,'audience':'healthcare','text':'Adequate staffing and fatigue management help maintain services during prolonged crises.'}
        a=self.ns['render_conversational_guidance']({'name':'Uppsala'},'heat',[p])
        self.assertIn('general healthcare-system preparedness priorities',a)
        self.assertIn('does not provide a hazard-specific clinical protocol',a);self.assertIn(p['text'],a)
    def test_generator_duplicate_keeps_conditions(self):
        a=['Use generators outside only.','Prepare for power outages.','Only use generators and grills outdoors and away from windows.']
        result=self.ns['deduplicate_guidance'](a)
        self.assertEqual(len(result),2);self.assertIn(a[1],result);self.assertIn(a[2],result)
    def test_auth_error_actionable_without_secret(self):
        self.ns['requests'].get.side_effect=None;self.ns['requests'].get.return_value=SimpleNamespace(status_code=401)
        r=self.ns['fetch_weathermetro_data'](59.8,17.6,'Uppsala');a=self.ns['weather_unavailable_message'](r)
        self.assertEqual(r['reason'],'http_401');self.assertIsNone(r['temp'])
        self.assertIn('API credentials',a);self.assertIn('restart the app',a);self.assertNotIn('offline-test-key',a)

    def test_policy_heading_removed_without_rewriting_sentence(self):
        raw='A RESILIENT HEALTH SYSTEM BUILDS ON\nMULTIPLE DIMENSIONS\nClimate-resilient health systems must be able to manage shocks through prevention, surge response, and post-shock recovery.'
        sentences=self.ns['complete_policy_sentences'](raw)
        self.assertEqual(sentences,['Climate-resilient health systems must be able to manage shocks through prevention, surge response, and post-shock recovery.'])
    def test_healthcare_hypothetical_keeps_scenario_label(self):
        p={'source':'CLIMES_Healthcare_preparedness.pdf','page':4,'audience':'healthcare','text':'Adequate staffing and fatigue management help maintain services during prolonged crises.'}
        a=self.ns['render_conversational_guidance']({'name':'Uppsala'},'heat',[p],{'temperature_c':32,'duration_days':3})
        self.assertIn('hypothetical scenario, not a confirmed forecast',a)
    def test_acceptance_rejects_ignored_month_even_if_count_matches(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('conversation_acceptance',ROOT/'climes_current/START.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        events=[{'event':'route','route':'historical','location':{'name':'Uppsala'},
                 'detail':{'temperature_c':31,'duration_days':3,'month':7}},
                {'event':'historical_tool','result':{'query_days':3,'minimum_duration_used':3,'threshold_used':31}}]
        status,_=module.validate_case(self.ns,'historical July','irrelevant',events)
        self.assertEqual(status,'FAIL')
    def test_real_index_policy_sentences_have_context(self):
        import sqlite3
        con=sqlite3.connect(f'file:{ROOT}/knowledge/chroma.sqlite3?mode=ro',uri=True)
        docs=[]
        for ident,source in con.execute("select id,string_value from embedding_metadata where key='source' and string_value like '%CLIMES_Healthcare_preparedness.pdf'").fetchall():
            raw=con.execute("select string_value from embedding_metadata where id=? and key='chroma:document'",(ident,)).fetchone()[0]
            docs.append((SimpleNamespace(page_content=raw,metadata={'source':source}),.9))
        con.close()
        store=Mock();store.similarity_search_with_relevance_scores.return_value=docs
        self.ns['Chroma']=Mock(return_value=store);self.ns['embeddings']=object()
        with patch.object(self.ns['os'].path,'exists',return_value=True): selected=self.ns['retrieve_manual_passages']('heat',healthcare=True)
        text='\n'.join(p['text'] for p in selected)
        self.assertIn('Adequate staffing',text)
        self.assertIn('medicines, protective equipment',text)
        self.assertFalse(any(line.startswith('This requires') for line in text.splitlines()))
        self.assertNotIn('A RESILIENT HEALTH SYSTEM BUILDS ON',text)
        for p in selected:
            raw=' '.join(p['raw_text'].split())
            for line in p['text'].splitlines(): self.assertIn(line,raw)

if __name__=='__main__': unittest.main(verbosity=2)
