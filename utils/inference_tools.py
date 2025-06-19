import torch
import torch.distributed as dist
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'

def check_previous_results(dataset, prev_file, out_file):
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))
    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without do anything
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return None, None, None, None
    
    data = data[~data['index'].isin(res)]
    lt = len(data)
    return data, lt, res, data_indices
    

def infer_data_local(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False
):
    from ipdb import set_trace as st
    st()
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if 'Failed to obtain answer via API' not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    # do inference
    data, lt, res, data_indices = check_previous_results(dataset, prev_file, out_file)
    if data is None and lt is None and res is None:
        return None
    
    model.set_dump_image(dataset.dump_image)
    for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.error(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)
    res = {k: res[k] for k in data_indices}
    dump(res, out_file)

    if world_size > 1:
        dist.barrier()

    #gather results from all ranks
    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model


def infer_data_api(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False
):
    from ipdb import set_trace as st
    st()
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if 'Failed to obtain answer via API' not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    # do inference
    data, lt, res, data_indices = check_previous_results(dataset, prev_file, out_file)
    if data is None and lt is None and res is None:
        return None
    
    indices = list(data['index'])
    index_set=set(indices)

    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices_api = len(data), list(data['index'])
    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file_api = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    res_api = {}
    if osp.exists(out_file_api):
        res_api = load(out_file_api)
        if ignore_failed:
            res_api = {k: v for k, v in res_api.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices_api, structs) if i not in res_api]
    indices_api = [i for i in indices_api if i not in res_api]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file_api, keys=indices_api)

    res_api = load(out_file_api)
    if index_set is not None:
        res_api = {k: v for k, v in res_api.items() if k in index_set}
    os.remove(out_file_api)


    for idx in indices:
        assert idx in res_api
    res.update(res_api)
    res = {k: res[k] for k in data_indices}
    dump(res, out_file)

    if world_size > 1:
        dist.barrier()

    #gather results from all ranks
    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model