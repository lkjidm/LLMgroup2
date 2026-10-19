"""Offline integration contracts; no model, HTTP, speech, or live vector search."""
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.argv = [str(ROOT/'climes_current/check_final.py')]
legacy = runpy.run_path(str(ROOT/'climes_current/check_final.py'), run_name='integration_helpers')
import heatwave_stats as stats


class IntegrationChecks(unittest.TestCase):
    def setUp(self):
        self.ns = legacy['helpers']()

    def test_exact_query_not_nearby_category(self):
        data = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=10),
                             'tmax': [30,30,30,0,31,31,31,31,31,0]})
        with patch.object(stats, '_df', stats._prepare(data)):
            exact = stats.historical_heatwave_stats(4,31)
            nearby = stats.historical_heatwave_stats(3,30)
        self.assertEqual(exact['historical_count'], 1)
        self.assertEqual(nearby['historical_count'], 2)
        self.assertEqual(exact['threshold_used'], 31)
        self.assertEqual(exact['minimum_duration_used'], 4)
        self.assertEqual(exact['longest_similar']['duration'], 5)

    def test_gaps_and_missing_temperatures_break_runs(self):
        data = pd.DataFrame({'date': ['2020-01-01','2020-01-02','2020-01-04','2020-01-05'], 'tmax': [32]*4})
        self.assertEqual(len(stats.find_heatwaves(data,31,3)),0)
        data = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=5), 'tmax': [32,32,None,32,32]})
        self.assertEqual(len(stats.find_heatwaves(data,31,3)),0)

    def test_order_and_duplicate_dates(self):
        data = pd.DataFrame({'date': ['2020-01-03','2020-01-01','2020-01-02'], 'tmax': [32]*3})
        self.assertEqual(len(stats.find_heatwaves(data,31,3)),1)
        with self.assertRaises(ValueError): stats.find_heatwaves(pd.concat([data,data]),31,3)

    def test_invalid_inputs_and_unavailable_csv(self):
        for days, temp in [(True,32),(0,32),(3.5,32),(3,True),(3,float('nan')),(3,float('inf'))]:
            self.assertIn('error', stats.historical_heatwave_stats(days,temp))
        with patch.object(stats, '_df', None), patch.object(stats, '_CSV', Path('/nonexistent/climes.csv')):
            self.assertIn('error', stats.historical_heatwave_stats(4,31))

    def test_bundled_series_and_independent_run_count(self):
        result = stats.historical_heatwave_stats(4,31)
        data = pd.read_csv(ROOT/'uppsala_tmax_1985_2026.csv')
        count, length, previous = 0, 0, None
        for row in data.itertuples():
            date = pd.Timestamp(row.date)
            if previous is None or (date-previous).days != 1 or not pd.notna(row.tmax) or row.tmax < 31:
                count += length >= 4
                length = 0
            if pd.notna(row.tmax) and row.tmax >= 31: length += 1
            previous = date
        count += length >= 4
        self.assertEqual(result['historical_count'], count)
        self.assertEqual(result['query_temp'],31)
        self.assertEqual(result['query_days'],4)
        self.assertGreater(result['valid_observation_days'],14000)

    def test_history_route_without_weather_or_selector(self):
        forbid = Mock(side_effect=AssertionError('Wrong service'))
        self.ns['retrieve_manual_passages'] = self.ns['generate_passage_selection'] = forbid
        answer = self.ns['build_answer']('Historically how often has Uppsala had 4 consecutive days at 31°C?')
        self.assertIn('at least 31°C for at least four consecutive days', answer)
        self.assertIn('does not determine the health risk or predict a future event', answer)
        self.ns['requests'].get.assert_not_called()
        forbid.assert_not_called()

    def test_conditional_duration_route_and_celsius_conversion(self):
        route, detail = self.ns['classify_request']('What is the risk if Uppsala has 4 days at 31°C?')
        self.assertEqual(route,'hypothetical')
        self.assertEqual(detail['duration_days'],4)
        route, detail = self.ns['classify_request']('How often historically: 3 days at 95°F?')
        self.assertEqual(route,'historical')
        self.assertAlmostEqual(detail['temperature_c'],35)

    def test_no_invented_duration_for_tomorrow(self):
        self.ns['get_historical_context'] = Mock(side_effect=AssertionError('No duration supplied'))
        self.ns['retrieve_manual_passages'] = lambda topic: []
        answer = self.ns['build_answer']('Suppose it is 32°C in Uppsala tomorrow. What precautions should I take?')
        self.assertIn('32°C',answer)
        self.ns['get_historical_context'].assert_not_called()
        for text in ['historical frequency at 32°C', 'historical 0 days at 32°C', 'historical -4 days at 32°C', 'historical 2.5 days at 32°C']:
            self.assertIn(self.ns['classify_request'](text)[0], {'clarify_history','clarify_scenario'})

    def test_hypothetical_history_combines_with_excerpts(self):
        self.ns['retrieve_manual_passages'] = lambda topic: []
        answer = self.ns['build_answer']('What is the risk if Uppsala has 4 days at 31°C?')
        self.assertIn('user-supplied assumption',answer)
        self.assertIn('bundled Uppsala station records',answer)
        self.assertNotIn('current weather observation',answer.casefold())
        self.ns['requests'].get.assert_not_called()

    def test_local_and_healthcare_sources_are_retrievable(self):
        def doc(name, text): return SimpleNamespace(page_content=text,metadata={'source':'knowledge\\'+name,'page':0})
        docs = [(doc('30012.pdf','HANDBOK I KOMMUNAL KRISBEREDSKAP Värmeböljor och höga temperaturer'),.99),
                (doc('CLIMES_Healthcare_preparedness.pdf','A resilient health system requires adequate staffing, secure essential supplies and strategic reserves to maintain services during prolonged climate-related crises. Preparedness training should support healthcare staff before, during and after shocks.'),.95),
                (doc('uppsala-tatort-a1.pdf','map'),1),
                (doc('ready.gov_extreme-heat_hazard-info-sheet.pdf','Avoid strenuous activities. Wear light clothing. Drink plenty of fluids.'),.8),
                (doc('30076.pdf','Flood passage, wrong topic'),1)]
        store=Mock(); store.similarity_search_with_relevance_scores.return_value=docs
        self.ns['Chroma']=Mock(return_value=store);self.ns['embeddings']=object()
        with patch.object(self.ns['os'].path,'exists',return_value=True):
            residents=self.ns['retrieve_manual_passages']('heat')
            healthcare=self.ns['retrieve_manual_passages']('heat',healthcare=True)
        self.assertEqual(residents[0]['source'],'ready.gov_extreme-heat_hazard-info-sheet.pdf')
        self.assertNotIn('CLIMES_Healthcare_preparedness.pdf',[p['source'] for p in residents])
        self.assertEqual(healthcare[0]['source'],'CLIMES_Healthcare_preparedness.pdf')
        self.assertNotIn('30076.pdf',[p['source'] for p in healthcare])
        self.assertTrue(all('filter' in call.kwargs for call in store.similarity_search_with_relevance_scores.call_args_list))
        answer=self.ns['render_manual_answer']({'name':'Uppsala'},'heat',healthcare)
        self.assertIn('not personalised resident guidance',answer)
        self.assertNotIn('HANDBOK',answer)
        self.assertIn('p. 1',answer)

    def test_conversational_answer_has_actions_without_dumping_pdf(self):
        passages=[{'id':'P1','source':'ready.gov_extreme-heat_hazard-info-sheet.pdf','page':1,
                   'audience':'residents','text':'If you’re outside, find shade.\nWear light clothing.'},
                  {'id':'P2','source':'ready.gov_extreme-heat_hazard-info-sheet.pdf','page':1,
                   'audience':'residents','text':'Wear light clothing.\nAvoid strenuous activities.'}]
        answer=self.ns['render_manual_answer']({'name':'Uppsala'},'heat',passages,{'temperature_c':32,'topic':'heat'})
        self.assertIn('For your assumed 32°C in Uppsala',answer)
        self.assertIn('not a confirmed forecast',answer)
        self.assertIn('- If you’re outside, find shade.',answer)
        self.assertEqual(answer.count('Wear light clothing.'),1)
        self.assertNotIn('.pdf',answer)
        self.assertNotIn('source excerpts',answer)
        self.assertNotIn('> ',answer)
        self.assertNotIn('35°C',answer)
        self.assertIn('not Swedish official advice',answer)

    def test_healthcare_truncated_fragments_are_not_displayed(self):
        raw='be treated as a core security capability.\nAdequate staffing, psychosocial support\nand fatigue management help maintain\nservices during prolonged crises and\nreduce the risk of burnout and turnover.\nSurge response also relies on secure\naccess to medicines, protective equipment\nand essential medical consumables. This\nrequires resilient supply chains and\nstrategic reserves able to function during\ndisruptions. The COVID-19 pandemic\nshowed how rapidly shortages can emerge'
        sentences=self.ns['complete_policy_sentences'](raw)
        self.assertGreaterEqual(len(sentences),2)
        self.assertTrue(all(sentence.endswith('.') for sentence in sentences))
        self.assertTrue(any('Adequate staffing' in sentence for sentence in sentences))
        self.assertFalse(any('pandemic' in sentence or sentence.startswith('be treated') for sentence in sentences))

    def test_cover_and_reference_list_are_not_precautions(self):
        for raw in ['HANDBOK I KOMMUNAL KRISBEREDSKAP 4. RISKKATALOG Värmeböljor och höga temperaturer',
                    'Läs mer Varningar och meddelanden (smhi.se) Meteoalarm tillhandahåller relevant information',
                    'Key References. Visit www.example.com for more information about heat.']:
            self.assertEqual(self.ns['resident_action_sentences'](raw,'heat'),[])
            self.assertFalse(self.ns['is_substantive_policy_text'](raw))

    def test_pdf_wrap_and_full_diet_caveat_preserved(self):
        raw='Wear loose, lightweight, light-\ncolored clothing.\nDrink plenty of fluids to stay hydrated. If you or someone you care\nfor is on a special diet, ask a doctor what would be best.'
        actions=self.ns['resident_action_sentences'](raw,'heat')
        self.assertEqual(len(actions),2)
        self.assertIn('ask a doctor what would be best.',actions[-1])
        incomplete='Drink plenty of fluids to stay hydrated. If you or someone you care for is on a special diet,'
        self.assertEqual(self.ns['resident_action_sentences'](incomplete,'heat'),[])

    def test_one_action_is_not_sufficient(self):
        passages=[{'id':'P1','text':'Wear light clothing.','audience':'residents'}]
        self.ns['generate_passage_selection']=lambda *args:'{"selected_ids":["P1"]}'
        self.assertEqual(self.ns['select_passages']('heat',passages),([], 'insufficient_actions'))

    def test_real_index_contains_usable_english_actions_for_all_hazards(self):
        import sqlite3
        con=sqlite3.connect(f'file:{ROOT}/knowledge/chroma.sqlite3?mode=ro',uri=True)
        docs=[]
        for ident,source in con.execute("select id,string_value from embedding_metadata where key='source'").fetchall():
            raw=con.execute("select string_value from embedding_metadata where id=? and key='chroma:document'",(ident,)).fetchone()[0]
            docs.append((SimpleNamespace(page_content=raw,metadata={'source':source}),.9))
        con.close()
        store=Mock();store.similarity_search_with_relevance_scores.return_value=docs
        self.ns['Chroma']=Mock(return_value=store);self.ns['embeddings']=object()
        with patch.object(self.ns['os'].path,'exists',return_value=True):
            for topic in ('heat','cold','flood'):
                passages=self.ns['retrieve_manual_passages'](topic)
                actions={line for p in passages for line in p['text'].splitlines()}
                self.assertGreaterEqual(len(actions),2,topic)
                self.assertTrue(all(p['source']==self.ns['manual_for_topic'](topic) for p in passages))
                self.assertNotIn('HANDBOK',' '.join(actions))

    def test_known_bad_screenshot_cannot_pass_acceptance(self):
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('acceptance_quality',ROOT/'climes_current/START.py')
        module=module_from_spec(spec);spec.loader.exec_module(module)
        passage={'id':'P1','source':'30012.pdf','text':'HANDBOK I KOMMUNAL KRISBEREDSKAP','raw_text':'HANDBOK I KOMMUNAL KRISBEREDSKAP'}
        location={'name':'Uppsala','found':True}
        detail={'topic':'heat','temperature_c':32}
        events=[{'event':'route','route':'hypothetical','location':location,'detail':detail},
                {'event':'retrieval','passages':[passage]},
                {'event':'evidence_selection','selected_ids':['P1'],'method':'retrieval_fallback'}]
        answer=self.ns['render_manual_answer'](location,'heat',[passage],detail)
        self.assertEqual(module.validate_case(self.ns,'Suppose it is 32°C in Uppsala tomorrow.',answer,events)[0],'FAIL')

    def test_healthcare_routing_and_unavailable_history(self):
        self.assertTrue(self.ns['classify_request']('What healthcare preparedness is recommended for extreme heat?')[1]['healthcare'])
        text=self.ns['format_historical_reply']({'error':'Historical observations unavailable'}, {})
        self.assertIn('cannot say how often this occurred',text)

    def test_short_history_uses_actual_query_and_count(self):
        for days,temp,count,expected in [(4,31,2,'there were two separate runs'),
                                         (3,30,1,'there was one separate run'),
                                         (1,35,0,'no matching runs were found'),
                                         (12,27.5,14,'there were 14 separate runs')]:
            with self.subTest(days=days,temp=temp,count=count):
                result={'historical_count':count,'query_days':days,'query_temp':temp,
                        'period':'1990-01-01 to 2020-12-31'}
                answer=self.ns['format_historical_reply'](result,{'name':'Flogsta'})
                self.assertIn(expected,answer)
                self.assertIn(f'at least {temp:g}°C',answer)
                self.assertIn('1990–2020',answer)
                self.assertNotIn('Flogsta station',answer)
                self.assertNotIn('empirical_rate_per_year',answer)
                self.assertNotIn('**',answer)
                self.assertNotIn('\n',answer)
                self.assertLess(len(answer.split()),85)


if __name__ == '__main__':
    unittest.main(verbosity=2)
