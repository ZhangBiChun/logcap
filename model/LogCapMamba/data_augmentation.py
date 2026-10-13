import random
from typing import Dict, List, Optional, Sequence, Tuple

# THNS24 event CSV (`preprocess_thns24_event_level.py`) encodes each event as
#   (act_id, hour_of_day, sem_id)
# where act_id is CERT channel (logon/http/email/file/connect/...),
# hour_of_day is `datetime.hour` in **0..23** (not a generic bin index),
# and sem_id comes from `SemanticFeatureExtractor` (Http_Leak, File_Exe, ...).
# Real insider-related deviations in this pipeline are mostly: unusual semantics,
# off-hours / shifted usage, short bursts, or alien fragments — not random
# local time-reversal of a few unrelated events (which the CSV never synthesizes).


def augment_session(
    session: Sequence[Tuple[int, int, int]],
    vocab_sizes: Tuple[int, int, int],
    valid_sems_per_act: Optional[Dict[int, Sequence[int]]] = None,
    normal_session_pool: Optional[List[Sequence[Tuple[int, int, int]]]] = None,
    pool_exclude_idx: Optional[int] = None,
    max_len: Optional[int] = None,
    active_aug_ops: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """
    Build pseudo-anomalies from normal (act, hour, sem) sequences.
    Returns: (augmented_session, mask)
    mask is 1 for mutated/inserted events, 0 for normal events.
    """
    session = list(session)
    n_orig = len(session)
    if n_orig < 5:
        return session, [0] * n_orig

    session_with_mask = [(s[0], s[1], s[2], 0) for s in session]

    # Insider-style: 1-2 subtle edits
    available_ops = active_aug_ops if active_aug_ops is not None else [1, 2, 3, 4]
    if not available_ops:
        return [s[:3] for s in session_with_mask], [0] * len(session_with_mask)
        
    num_ops = random.randint(1, min(2, len(available_ops)))
    ops = random.sample(available_ops, k=num_ops)

    sem_vocab = int(vocab_sizes[2])

    for op in ops:
        n = len(session_with_mask)
        if op == 1:
            # Op 1: Act / Semantic Shift
            num_swaps = random.randint(1, 5)
            start_idx = random.randint(0, max(0, n - num_swaps))
            for i in range(start_idx, min(n, start_idx + num_swaps)):
                a, h, s, _ = session_with_mask[i]
                
                # 30% chance to completely change the Act (Channel)
                if random.random() < 0.3:
                    new_a = random.choice([1, 2, 3, 4, 5, 6, 7]) # All possible THNS24 acts
                    # If act changes, pick a valid semantic for the new act
                    if valid_sems_per_act and new_a in valid_sems_per_act and valid_sems_per_act[new_a]:
                        new_s = random.choice(valid_sems_per_act[new_a])
                    else:
                        new_s = 0
                    session_with_mask[i] = (new_a, int(h), int(new_s), 1)
                else:
                    # 70% chance to just change Semantic (original behavior)
                    if random.random() < 0.4 and valid_sems_per_act and a in valid_sems_per_act:
                        cand = list(valid_sems_per_act[a])
                        if cand:
                            session_with_mask[i] = (a, int(h), int(random.choice(cand)), 1)
                    else:
                        session_with_mask[i] = (a, int(h), random.randint(0, max(0, sem_vocab - 1)), 1)

        elif op == 2:
            num_dups = random.randint(5, 15)
            idx = random.randint(0, n - 1)
            event_to_dup = session_with_mask[idx]
            event_to_dup = (event_to_dup[0], event_to_dup[1], event_to_dup[2], 1)
            for _ in range(num_dups):
                session_with_mask.insert(idx, event_to_dup)
            n = len(session_with_mask)

        elif op == 3:
            w = random.randint(3, min(10, n))
            start = random.randint(0, n - w)
            off_hours = [0, 1, 2, 3, 4, 5, 22, 23]
            chosen_hour = random.choice(off_hours)
            for i in range(start, start + w):
                a, h, s, _ = session_with_mask[i]
                session_with_mask[i] = (a, chosen_hour, s, 1)

        elif op == 4:
            if not normal_session_pool or len(normal_session_pool) < 2:
                continue
            indices = list(range(len(normal_session_pool)))
            if pool_exclude_idx is not None and pool_exclude_idx in indices:
                indices.remove(pool_exclude_idx)
            if not indices:
                continue
            src_i = random.choice(indices)
            src = list(normal_session_pool[src_i])
            if len(src) < 5:
                continue
            
            frag_len = random.randint(5, min(15, len(src)))
            src_start = random.randint(0, len(src) - frag_len)
            fragment = [(x[0], x[1], x[2], 1) for x in src[src_start:src_start + frag_len]]
            
            dst_start = random.randint(0, n)
            for k in reversed(range(frag_len)):
                session_with_mask.insert(dst_start, fragment[k])
            n = len(session_with_mask)

    if max_len is not None and len(session_with_mask) > max_len:
        session_with_mask = session_with_mask[-max_len:]

    session = [(x[0], x[1], x[2]) for x in session_with_mask]
    mask = [x[3] for x in session_with_mask]
    return session, mask
