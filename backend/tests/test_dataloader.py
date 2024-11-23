from src.model.dataloader import MovieDataloader


def test_singleton_feature():
    inst1 = MovieDataloader()
    inst2 = MovieDataloader()
    inst3 = MovieDataloader()
    assert inst1 == inst2 and inst1 == inst3 and inst2 == inst3, \
        "The dataloader does not follow the singleton object pattern. Please revisit the implementation."
