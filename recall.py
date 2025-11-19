import torch
import tqdm
import gc
import logging

logging.basicConfig(level=logging.INFO)


def get_sim(txt_feature, cld_feature, sim_func, t2c=True, txt_block_size=100, cld_block_size=100):
    """
        get similarities
    :param cld_block_size:   cld_num per block
    :param txt_block_size:   txt_num per block
    :param sim_func:       ([bb1, ss1, h1], [bb2, ss2, h2]) -> [bb1, bb2, h2]
    :param txt_feature:    b1, s1, h1
    :param cld_feature:    b2, s2, h2
    :return:               [b1, b2]
    """
    logging.info("Starting get_sim")
    b1, s1, h1 = txt_feature.shape
    b2, s2, h2 = cld_feature.shape

    sims = torch.zeros(b1, b2, device=txt_feature.device)

    for txt_blk_idx_start in tqdm.trange(0, b1, txt_block_size):    # tqdm.trange
        for cld_blk_idx_start in range(0, b2, cld_block_size):

            txt_blk = txt_feature[txt_blk_idx_start: txt_blk_idx_start + txt_block_size]
            cld_blk = cld_feature[cld_blk_idx_start: cld_blk_idx_start + cld_block_size]

            with torch.no_grad():
                sim = sim_func(txt_blk, cld_blk)

            sims[
                txt_blk_idx_start: txt_blk_idx_start + txt_block_size,
                cld_blk_idx_start: cld_blk_idx_start + cld_block_size
            ] = sim

            del txt_blk, cld_blk, sim
            gc.collect()
            # check_gpu_memory()
    logging.info("Finished get_sim")
    return sims


def get_rank(txt_feature, cld_feature, link, sim_func, t2c=True, txt_block_size=100, cld_block_size=100):
    """
        get rank
    :param t2c:              text to cloud (True) or cloud to text (False)
    :param link:             [[txt_i, cld_i], ...]
    :param txt_feature:      see get_sim
    :param cld_feature:      see get_sim
    :param sim_func:         see get_sim
    :param txt_block_size:   see get_sim
    :param cld_block_size:   see get_sim
    :return:  b1 or b2
    """
    logging.info("Starting get_rank")
    sims = get_sim(txt_feature, cld_feature, sim_func, t2c, txt_block_size, cld_block_size)
    if not t2c:
        sims = sims.transpose(1, 0)
        link = [[r[1], r[0]] for r in link]

    ranks = torch.full((sims.shape[0],), -1, device=txt_feature.device)
    for r in tqdm.tqdm(link):
        i1, i2 = r
        tgt_sim = sims[i1, i2]
        rank = torch.sum(sims[i1] > tgt_sim).item() + 1
        if ranks[i1] == -1 or (ranks[i1] > -1 and ranks[i1] > rank):
            ranks[i1] = rank

    have_rank = torch.nonzero(ranks > -1).squeeze()
    if len(have_rank) < sims.shape[0]:
        import warnings
        warnings.warn(
            f"There exists {'text' if t2c else 'cloud'} having no matched {'cloud' if t2c else 'text'} in link"
        )
        ranks = ranks[have_rank]

    del sims, have_rank
    gc.collect()
    # check_gpu_memory()
    logging.info("Finished get_rank")
    return ranks


def evaluation(txt_feature, cld_feature, link, sim_func, t2c=True, txt_block_size=50, cld_block_size=50, recall_idx=None):
    """
        get recall and mean rank
    :param recall_idx:       [recall_{i}, ...]
    :param txt_feature:    see get_rank
    :param cld_feature:    see get_rank
    :param link:           see get_rank
    :param sim_func:       see get_rank
    :param t2c:            see get_rank
    :param txt_block_size: see get_rank
    :param cld_block_size: see get_rank
    :return:   [r_{i}], mr, mrr
    """
    if recall_idx is None:
        recall_idx = [1, 5, 10]
    rank = get_rank(txt_feature, cld_feature, link, sim_func, t2c, txt_block_size, cld_block_size)
    recalls = []

    logging.info("calculating recalls")
    for recall_i in tqdm.tqdm(recall_idx):
        recalls.append(
            torch.mean((rank <= recall_i).float()).item()
        )

    mr = torch.mean(rank.float()).item()
    mrr = torch.mean(1.0 / rank.float()).item()

    del rank
    gc.collect()
    # check_gpu_memory()

    return recalls, mr, mrr


def check_gpu_memory():
    """

    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
