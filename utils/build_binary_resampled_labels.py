import argparse
import copy
import json
import random
from pathlib import Path


def load_items(input_path: Path):
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError(f'Unsupported JSON top-level type: {type(data).__name__}')



def sample_items(items, target_count: int, rng: random.Random):
    if target_count < 0:
        raise ValueError(f'target_count must be >= 0, got {target_count}')
    if target_count == 0:
        return [], False
    if not items:
        raise ValueError('Cannot sample from an empty class bucket')

    if target_count <= len(items):
        sampled = rng.sample(items, target_count)
        return [copy.deepcopy(item) for item in sampled], False

    sampled = [copy.deepcopy(rng.choice(items)) for _ in range(target_count)]
    return sampled, True



def main():
    parser = argparse.ArgumentParser(
        description='Build a new binary label JSON by controlling the number of class-0 and class-1 samples.'
    )
    parser.add_argument('--input_json', type=str, required=True, help='Path to the original label JSON file')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save the new label JSON file')
    parser.add_argument('--target_key', type=str, required=True, help='Binary target key to rebalance, e.g. FTCPTC')
    parser.add_argument('--num_class0', type=int, required=True, help='Number of class-0 samples to keep or generate')
    parser.add_argument('--num_class1', type=int, required=True, help='Number of class-1 samples to keep or generate')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed for reproducible sampling')
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    if not input_path.exists():
        raise FileNotFoundError(f'Input JSON not found: {input_path}')

    rng = random.Random(args.seed)
    items = load_items(input_path)

    class0_items = []
    class1_items = []
    ignored_items = []

    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Each JSON item must be a dict')
        label = item.get(args.target_key, -1)
        if label == 0:
            class0_items.append(item)
        elif label == 1:
            class1_items.append(item)
        else:
            ignored_items.append(item)

    if not class0_items:
        raise ValueError(f'No class-0 items found for target_key={args.target_key}')
    if not class1_items:
        raise ValueError(f'No class-1 items found for target_key={args.target_key}')

    sampled_class0, class0_used_replacement = sample_items(class0_items, args.num_class0, rng)
    sampled_class1, class1_used_replacement = sample_items(class1_items, args.num_class1, rng)

    output_items = sampled_class0 + sampled_class1
    rng.shuffle(output_items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(output_items, f, ensure_ascii=False, indent=2)

    print(f'Input JSON: {input_path}')
    print(f'Output JSON: {output_path}')
    print(f'Target key: {args.target_key}')
    print(f'Seed: {args.seed}')
    print('')
    print('Source distribution:')
    print(f'  class 0: {len(class0_items)}')
    print(f'  class 1: {len(class1_items)}')
    print(f'  ignored (not 0/1): {len(ignored_items)}')
    print('')
    print('Requested distribution:')
    print(f'  class 0: {args.num_class0}')
    print(f'  class 1: {args.num_class1}')
    print('')
    print('Sampling mode:')
    print(f'  class 0 replacement: {class0_used_replacement}')
    print(f'  class 1 replacement: {class1_used_replacement}')
    print('')
    print(f'Saved {len(output_items)} records.')


if __name__ == '__main__':
    main()
