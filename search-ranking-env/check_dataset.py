from datasets import load_dataset

def main():
    dataset = load_dataset("ms_marco", "v1.1", split="train[:0.1%]")
    print(dataset.features)
    for i in range(2):
        print(f"Sample {i}:", dataset[i])

if __name__ == "__main__":
    main()
