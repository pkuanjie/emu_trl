import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import logging
from pdb import set_trace as bp
import logging

logging.basicConfig(level=logging.WARNING)


def find_all_files(root, suffix=".jpg"):
    res = []
    for root, _, files in os.walk(root):
        for file in files:
            if file.endswith(suffix):
                res.append(os.path.join(root, file))
    return res


def print_rank0(info_str, level=logging.INFO):
    logging.log(level, info_str)


class CapDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
        super().__init__()
        self.captions = torch.load(os.path.join(data_dirs, "captions.th"))
        self.data = self.load_data(data_dirs)
        self.image_processor, self.text_processor, self.cross_image_processor = (
            image_processor,
            text_processor,
            cross_image_processor,
        )

    def process_img(self, img):
        logging.info(f"img shape: {img.size}")
        img_dict = {"vision": self.image_processor(img)}
        logging.info(f"img_dict shape: {img_dict['vision'].shape}")
        if self.cross_image_processor:
            img_dict.update({"cross": self.cross_image_processor(img)})
        return img_dict

    def process_text(self, answer):
        # return self.text_processor(answer, prompt)
        logging.info(f"answer: {answer}")
        text_dict = {"prompt": answer}
        return text_dict

    def load_data(self, data_dir):
        all_files = find_all_files(os.path.join(data_dir, "images"), suffix=".jpg")
        all_files = [file for file in all_files if file.split("/")[-1].split(".")[0] in self.captions.keys()]
        # names = [file.split('/')[-1].split('.')[0] for file in all_files]
        # names = [n for n in names if n in self.captions.keys()]
        print_rank0(f"find {len(all_files)} samples in all...")
        return all_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # img
        try:
            img = Image.open(data).convert("RGB")
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        img_dict = self.process_img(img)
        # text
        label = data.split("/")[-1].split(".")[0]
        uni_key = label

        caption = self.captions[label]
        text_dict = self.process_text(caption)
        if text_dict is None:
            print_rank0(
                f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}",
                level=logging.WARNING,
            )
            return {}
        # other attr
        ret = {**img_dict, **text_dict, "question_id": uni_key}
        return ret


def cap_data_collate_fn(batch):
    imgs = torch.stack([x["vision"] for x in batch])
    texts = [x["prompt"] for x in batch]
    # input_ids = torch.stack([x["input_ids"] for x in texts])
    # attention_mask = torch.stack([x["attention_mask"] for x in texts])
    logging.info(f"imgs shape: {imgs.shape}")
    logging.info(f"texts len: {len(texts)}")
    return {
        "image": imgs,
        "prompt": texts,
        # "input_ids": input_ids,
        # "attention_mask": attention_mask,
    }


def get_loader(
    image_processor,
    text_processor,
    batch_size,
    num_workers,
    args,
    data_dirs,
    shuffle,
    drop_last,
    cross_image_processor=None,
    **kwargs,
):
    dataset = CapDataset(image_processor, text_processor, args, data_dirs, cross_image_processor, **kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=cap_data_collate_fn,
    )
    return loader
