# -*- coding: utf-8 -*-
"""MCP 全链路自检：v3 登记册 ↔ AI 工具端到端（离屏，独立端口 18799）。

与 ai_selftest（工具语义）/ ai_compute_test（计算通道）/ ai_codeact_test
（CodeAct）互补：本套专测**参数管理改造后的登记面**——

  · list_params 覆盖全部在册参数（含 external：lights/waves/psi/mode/decays）；
  · 显示类字段不入册（topn/readout/display）→ AI 不可见即不可改；
  · set_param 走工厂件 / 索引 combo / external 整表三条通道；
  · 非法值与未知参数被拒；
  · save_archive → 破坏 → load_archive 往返（含双原子模式与波形）；
  · undo 快照回滚；run_python 回传 result；journal 归因带 reason；
  · GUI 控件同步（人机等效）。

用法: D:\\python310\\python.exe -u checks\\ai_chain_selftest.py
"""
import asyncio
import json
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['EL_MCP_PORT'] = '18799'

from PyQt5.QtWidgets import QApplication, QMessageBox
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

import app_entry as main_mod

P, F = [], []
DONE = {}
WIN = None


def check(name, cond):
    (P if cond else F).append(name)
    print(f'  {"PASS" if cond else "FAIL"}  {name}')


async def _drive(port):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with streamable_http_client(f'http://127.0.0.1:{port}/mcp') as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            check('17 工具齐全', len(tools) >= 17)

            async def call(name, **kw):
                res = await s.call_tool(name, kw)
                return json.loads(res.content[0].text)

            # ---- list_params：v3 全量在册 ----
            lp = (await call('list_params'))['params']
            names = set(lp)
            need = {'field.B_gauss', 'field.E_vpcm', 'field.window',
                    'field.n_photon', 'field.lights', 'groups.list',
                    'dynamic.t_max', 'dynamic.npts', 'dynamic.method',
                    'dynamic.frozen_h_ns', 'dynamic.per_freq_M',
                    'dynamic.decay_index', 'dynamic.mode',
                    'dynamic.pair.R_um', 'dynamic.pair.theta_deg',
                    'dynamic.pair.phi_deg', 'dynamic.waves_single',
                    'dynamic.waves_A', 'dynamic.waves_B',
                    'dynamic.psi_single', 'dynamic.psi_A', 'dynamic.psi_B',
                    'dynamic.decays'}
            missing = need - names
            check('list_params 覆盖全部登记参数', not missing)
            if missing:
                print('    缺失:', missing)
            check('factory 件带元数据（B unit=G）',
                  lp['field.B_gauss'].get('unit') == 'G')
            check('显示类字段不入册（topn/readout/display）',
                  not any('topn' in n or 'readout' in n or 'display' in n
                          for n in names))

            # ---- 建基础场景 ----
            for n, l, j in ((60, 0, 0.5), (60, 1, 1.5), (6, 1, 1.5)):
                r = await call('add_level_group', n=n, l=l, j=j)
                check(f'add_level_group {n}{l}{j} ok', r.get('ok'))
            await call('set_param', path='dynamic.decay_index', value=0)

            # ---- set_param 标量（工厂件 / 索引 combo）----
            r = await call('set_param', path='field.B_gauss', value=30.0,
                           reason='链路自检改B')
            check('set_param B 标量 ok', r.get('ok'))
            r = await call('set_param', path='dynamic.method', value='hard')
            check('set_param method combo ok', r.get('ok'))
            r = await call('set_param', path='dynamic.decay_index', value=1)
            check('set_param decay_index 索引 ok', r.get('ok'))
            await call('set_param', path='dynamic.decay_index', value=0)
            r = await call('set_param', path='field.B_gauss', value='abc')
            check('set_param 非法值被拒', not r.get('ok'))
            r = await call('set_param', path='no.such.param', value=1)
            check('未知参数被拒+提示可用清单',
                  not r.get('ok') and 'field.B_gauss' in r.get('msg', ''))

            # ---- set_param external：lights ----
            lights = [{'frequency_hz': 6800e9, 'intensity': 0.05,
                       'polarization': 'π', 'scope': 'both', 'enabled': True}]
            r = await call('set_param', path='field.lights', value=lights,
                           reason='链路自检加光')
            check('set_param field.lights ok', r.get('ok'))
            st = await call('get_state', scope='field')
            check('lights 写入生效（1 行）',
                  len(st['params']['field.lights']) == 1)

            # ---- mode + waves + psi ----
            r = await call('set_param', path='dynamic.mode', value=1,
                           reason='切双原子')
            check('set_param dynamic.mode ok', r.get('ok'))
            r = await call('set_waves', atom='A', waves=[
                {'name': 'w', 'freq_mhz': 17236.743, 'intensity': 5.5e-5,
                 'pol': 'σ-', 'offset_ns': 0.0,
                 'env': ['gauss', 250, 40, 1.0]}], reason='链路自检设波')
            check('set_waves atom=A ok', r.get('ok'))
            st = await call('get_state', scope='dynamic')
            pk = st['params'].get('dynamic.psi_A')
            check('psi_A 在册（身份键）', 'dynamic.psi_A' in st['params'])
            if pk:
                r = await call('set_param', path='dynamic.psi_A', value=pk,
                               reason='链路自检设初态')
                check('set_param psi_A ok', r.get('ok'))

            # ---- get_state v3 形状 ----
            full = await call('get_state')
            check('get_state v3 扁平 params',
                  full.get('meta', {}).get('version') == 3
                  and isinstance(full.get('params'), dict))

            # ---- save/load 往返 ----
            r = await call('save_archive', path='save_state/_chain.json')
            check('save_archive ok', r.get('ok'))
            await call('set_param', path='field.B_gauss', value=1.0,
                       reason='链路自检破坏')
            await call('set_param', path='dynamic.mode', value=0)
            r = await call('load_archive', path=r['path'])
            check('load_archive ok', r.get('ok'))
            summ = await call('summary')
            check('load 后 B=30 / lights=1 / 双原子+波形',
                  abs(summ['B_gauss'] - 30.0) < 1e-6 and summ['n_lights'] == 1
                  and summ['two_body'] and summ['n_waves'] >= 1)

            # ---- undo ----
            before = (await call('summary'))['B_gauss']
            await call('set_param', path='field.B_gauss', value=7.7,
                       reason='待撤销')
            r = await call('undo', reason='链路自检撤销')
            after = (await call('summary'))['B_gauss']
            check('undo 回滚 B', r.get('ok') and abs(after - before) < 1e-6)

            # ---- run_python（CodeAct）----
            r = await call('run_python', code=(
                "set_param('field.E_vpcm', 12.5, reason='链路自检 codeact')\n"
                "s = summary()\n"
                "result = {'E': s.get('E_vpcm'), 'n': s.get('n_groups')}\n"),
                timeout=120.0)
            check('run_python ok 且 result 回传',
                  r.get('ok') and (r.get('result') or {}).get('E') == 12.5)

            # ---- get_activity 归因 ----
            act = await call('get_activity', since_seq=0)
            ai_ops = [o for o in act['ops']
                      if str(o.get('actor', '')).startswith('ai')]
            kinds = {o.get('kind') for o in ai_ops}
            check('journal 记 ai op（8+）', len(ai_ops) >= 8)
            check('含 set_param/run_python/load_archive 各型',
                  {'set_param', 'run_python', 'load_archive'} <= kinds)
            check('ai op 带 reason', any(o.get('reason') for o in ai_ops))

            # ---- GUI 控件同步（人机等效）----
            check('GUI B 控件=30', abs(WIN.field.b_spin.value() - 30.0) < 1e-6)
            check('GUI 光场表 1 行', WIN.field.table.rowCount() == 1)
            check('GUI 双原子模式', WIN.dynamic._is_two_body)
            check('GUI 原子A波形 1 条',
                  len(WIN.dynamic.wave_panel_A.waves) == 1)


def _client_thread(port):
    try:
        asyncio.run(_drive(port))
    except Exception:
        F.append('客户端异常:\n' + traceback.format_exc())
        traceback.print_exc()
    finally:
        DONE['__done__'] = True


def main():
    global WIN
    app = QApplication([])
    WIN = main_mod.MainWindow()
    WIN.show()
    app.processEvents()
    port = WIN._init_ai()
    from checks.checklib import (require_ai_server as _need_ai)
    _need_ai(port, 'ai_chain_selftest.py')
    check('MCP server 启动', bool(port))
    if not port:
        return 1
    t = threading.Thread(target=_client_thread, args=(port,), daemon=True)
    t.start()
    t0 = time.time()
    while not DONE.get('__done__') and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)
    print(f'\n通过 {len(P)} 项；失败 {len(F)} 项'
          + (f' ❌ {F}' if F else ' ✓'))
    try:
        os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'save_state', '_chain.json'))
    except OSError:
        pass
    return 1 if F else 0


if __name__ == '__main__':
    sys.exit(main())
