from src.model.dataloader import MovieDataloader


def test_singleton():
    dataloader_1 = MovieDataloader()
    dataloader_2 = MovieDataloader()
    assert dataloader_1 is dataloader_2
