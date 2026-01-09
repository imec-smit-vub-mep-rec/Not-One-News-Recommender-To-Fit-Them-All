import pytest
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sentence_transformers import SentenceTransformer  # Import to check model existence

from SentenceTransformerContentBased import SentenceTransformerContentBased

# --- Test Data ---

CONTENT_DICT = {
    0: "This is the content for item 0 about apples.",
    1: "Item 1 is about oranges and bananas.",
    2: "Another item, number 2, discussing apples again.",
    3: "Item 3 has unrelated content about cars.",
    # Item 4 has no content
    5: "Content for item 5, similar to apples."
}

# User 0 interacted with 0, 1
# User 1 interacted with 2, 3
# User 2 interacted with 4 (no content)
# User 3 interacted with 0, 5
# User 4 interacted with nothing
INTERACTION_MATRIX = csr_matrix([
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
])

NUM_USERS, NUM_ITEMS = INTERACTION_MATRIX.shape

# Use a smaller model for faster testing
TEST_MODEL = 'intfloat/multilingual-e5-large'
# TEST_MODEL = 'all-MiniLM-L6-v2' # Or use the default if needed

# Check if the test model exists locally or needs downloading
try:
    SentenceTransformer(TEST_MODEL)
    _model_exists = True
except Exception:
    _model_exists = False
    print(f"Warning: Test model '{TEST_MODEL}' not found. Skipping tests requiring encoding.")

# --- Fixtures ---

@pytest.fixture
def default_params():
    """Default parameters for the algorithm."""
    return {
        "content": CONTENT_DICT.copy(),
        "language": TEST_MODEL,
        "metric": "angular", # Using angular as it relates to cosine similarity
        "n_trees": 5, # Fewer trees for faster testing
        "num_neighbors": 3, # Fewer neighbors
        "verbose": False
    }

@pytest.fixture
def fitted_model(default_params):
    """A model instance fitted with default data."""
    if not _model_exists:
        pytest.skip(f"Test model '{TEST_MODEL}' not found.")
    model = SentenceTransformerContentBased(**default_params)
    model._fit(INTERACTION_MATRIX.copy())
    return model

# --- Test Cases ---

def test_initialization(default_params):
    """Test basic initialization and parameter setting."""
    model = SentenceTransformerContentBased(**default_params)
    assert model.content == default_params["content"]
    assert model.language == default_params["language"]
    assert model.metric == default_params["metric"]
    assert model.n_trees == default_params["n_trees"]
    assert model.num_neighbors == default_params["num_neighbors"]
    assert model.verbose == default_params["verbose"]
    assert model.embedding_dim is None # Input param
    assert model._embedding_dim is not None # Internal, inferred dim
    assert model.annoy_index is not None
    assert model.sentencetransformer is not None
    assert model._user_offset is None # Set during fit
    assert not model._users_in_annoy # Populated during fit
    assert not model._item_embeddings # Populated during fit


def test_initialization_with_embedding_dim(default_params):
    """Test initialization when embedding_dim is explicitly provided."""
    params = default_params.copy()
    # Get actual dim from model
    if not _model_exists:
        pytest.skip(f"Test model '{TEST_MODEL}' not found.")
    transformer = SentenceTransformer(params['language'])
    dummy_emb = transformer.encode("test")
    explicit_dim = dummy_emb.shape[0]
    params["embedding_dim"] = explicit_dim

    model = SentenceTransformerContentBased(**params)
    assert model.embedding_dim == explicit_dim
    assert model._embedding_dim == explicit_dim


@pytest.mark.skipif(not _model_exists, reason=f"Test model '{TEST_MODEL}' not found.")
def test_fit(default_params):
    """Test the _fit method."""
    model = SentenceTransformerContentBased(**default_params)
    model._fit(INTERACTION_MATRIX.copy())

    # Check fitted attributes
    assert model.n_users_ == NUM_USERS
    assert model.n_items_ == NUM_ITEMS
    assert model.n_features_in_ == NUM_ITEMS # Alias for n_items_
    assert model._user_offset == NUM_ITEMS + 100

    # Check item embeddings were created for items with content
    assert len(model._item_embeddings) == len([c for c in CONTENT_DICT if c < NUM_ITEMS])
    assert 0 in model._item_embeddings
    assert 1 in model._item_embeddings
    assert 2 in model._item_embeddings
    assert 3 in model._item_embeddings
    assert 4 not in model._item_embeddings # No content provided for item 4
    assert 5 in model._item_embeddings
    # Check embedding dimension consistency
    first_emb_dim = next(iter(model._item_embeddings.values())).shape[0]
    assert model._embedding_dim == first_emb_dim
    for emb in model._item_embeddings.values():
        assert emb.shape == (first_emb_dim,)

    # Check Annoy index
    assert model.annoy_index.get_n_items() == len(model._item_embeddings)
    # Ensure items added to Annoy are the ones with embeddings
    items_in_annoy = {model.annoy_index.get_item_vector(i)[0]: i for i in range(model.annoy_index.get_n_items())}
    assert set(items_in_annoy.keys()) == set(model._item_embeddings.keys())


@pytest.mark.skipif(not _model_exists, reason=f"Test model '{TEST_MODEL}' not found.")
def test_fit_no_content(default_params):
    """Test fitting when the content dictionary is empty."""
    params = default_params.copy()
    params["content"] = {}
    model = SentenceTransformerContentBased(**params)
    model._fit(INTERACTION_MATRIX.copy())

    assert model.n_users_ == NUM_USERS
    assert model.n_items_ == NUM_ITEMS
    assert len(model._item_embeddings) == 0
    assert model.annoy_index.get_n_items() == 0


@pytest.mark.skipif(not _model_exists, reason=f"Test model '{TEST_MODEL}' not found.")
def test_fit_items_without_content_in_matrix(default_params):
    """Test fit when interaction matrix contains items not in content dict."""
    # User 2 interacts with item 4, which has no content in CONTENT_DICT
    model = SentenceTransformerContentBased(**default_params)
    model._fit(INTERACTION_MATRIX.copy())

    assert model.n_users_ == NUM_USERS
    assert model.n_items_ == NUM_ITEMS
    assert 4 not in model._item_embeddings # Item 4 should not have an embedding
    assert model.annoy_index.get_n_items() == len(CONTENT_DICT) # Only items with content are indexed


@pytest.mark.skipif(not _model_exists, reason=f"Test model '{TEST_MODEL}' not found.")
def test_predict(fitted_model):
    """Test the _predict method."""
    predictions = fitted_model._predict(INTERACTION_MATRIX.copy())

    assert isinstance(predictions, csr_matrix)
    assert predictions.shape == (NUM_USERS, NUM_ITEMS)
    assert predictions.dtype == np.float32

    # Check specific user predictions
    # User 0 interacted with 0, 1. Should get recommendations based on apples/oranges/bananas.
    # Item 2 (apples) and 5 (apples) are likely candidates. Item 3 (cars) is unlikely.
    user0_preds = predictions.getrow(0).toarray().flatten()
    assert user0_preds[0] == 0 # Interacted item should not be recommended
    assert user0_preds[1] == 0 # Interacted item should not be recommended
    assert user0_preds[2] > 0 or user0_preds[5] > 0 # Expect recs for similar items 2 or 5
    if 3 in CONTENT_DICT: # Only check if item 3 has content
         assert user0_preds[3] < max(user0_preds[2], user0_preds[5]) # Item 3 (cars) should have lower score


    # User 1 interacted with 2 (apples), 3 (cars).
    user1_preds = predictions.getrow(1).toarray().flatten()
    assert user1_preds[2] == 0 # Interacted
    assert user1_preds[3] == 0 # Interacted
    assert user1_preds[0] > 0 or user1_preds[5] > 0 # Item 0/5 (apples) might be recommended

    # User 2 interacted with 4 (no content). Should get no personalized recs.
    user2_preds = predictions.getrow(2).toarray().flatten()
    assert np.all(user2_preds == 0)

    # User 3 interacted with 0, 5 (both apples). Should strongly recommend item 2 (apples).
    user3_preds = predictions.getrow(3).toarray().flatten()
    assert user3_preds[0] == 0 # Interacted
    assert user3_preds[5] == 0 # Interacted
    assert user3_preds[2] > 0 # Item 2 (apples) should be recommended

    # User 4 interacted with nothing. Should get no personalized recs.
    user4_preds = predictions.getrow(4).toarray().flatten()
    assert np.all(user4_preds == 0)

    # Ensure num_neighbors is respected (at most)
    for user_id in range(NUM_USERS):
        assert predictions.getrow(user_id).nnz <= fitted_model.num_neighbors


@pytest.mark.skipif(not _model_exists, reason=f"Test model '{TEST_MODEL}' not found.")
def test_predict_no_interactions(fitted_model):
    """Test prediction for a user matrix with no interactions."""
    empty_interactions = lil_matrix((NUM_USERS, NUM_ITEMS), dtype=INTERACTION_MATRIX.dtype)
    predictions = fitted_model._predict(empty_interactions.tocsr())

    assert isinstance(predictions, csr_matrix)
    assert predictions.shape == (NUM_USERS, NUM_ITEMS)
    assert predictions.nnz == 0 # No recommendations if no interaction history


@pytest.mark.skipif(not _model_exists, reason=f"Test model '{TEST_MODEL}' not found.")
def test_predict_error_before_fit(default_params):
    """Test that predict raises an error (or handles gracefully) if called before fit."""
    # Current implementation doesn't explicitly raise but might fail implicitly.
    # Let's check if it returns an empty matrix without erroring badly.
    model = SentenceTransformerContentBased(**default_params)
    # We expect predict to fail gracefully if annoy_index isn't built etc.
    # It should return an empty matrix without raising an unhandled exception.
    try:
        predictions = model._predict(INTERACTION_MATRIX.copy())
        assert isinstance(predictions, csr_matrix)
        assert predictions.shape == (NUM_USERS, NUM_ITEMS)
        assert predictions.nnz == 0
    except Exception as e:
        pytest.fail(f"_predict raised an unexpected exception before fit: {e}")


def test_distance_to_similarity(default_params):
    """Test the _distance_to_similarity conversion."""
    model = SentenceTransformerContentBased(**default_params)
    eps = 1e-6 # Epsilon for float comparisons

    # Angular / Dot (similar for normalized vectors, Annoy uses sqrt(2*(1-cos)))
    model.metric = 'angular'
    assert abs(model._distance_to_similarity(0.0) - 1.0) < eps  # cos = 1 -> dist = 0
    assert abs(model._distance_to_similarity(np.sqrt(2.0)) - 0.0) < eps # cos = 0 -> dist = sqrt(2)
    assert abs(model._distance_to_similarity(2.0) - (-1.0)) < eps # cos = -1 -> dist = 2
    # Test intermediate value (e.g., cos=0.5 -> dist=1)
    assert abs(model._distance_to_similarity(1.0) - 0.5) < eps

    model.metric = 'dot' # Assuming same behavior as angular for normalized vectors
    assert abs(model._distance_to_similarity(0.0) - 1.0) < eps
    assert abs(model._distance_to_similarity(1.0) - 0.5) < eps
    assert abs(model._distance_to_similarity(np.sqrt(2.0)) - 0.0) < eps

    # Euclidean
    model.metric = 'euclidean'
    assert abs(model._distance_to_similarity(0.0) - 1.0) < eps # d=0 -> sim=1
    assert model._distance_to_similarity(1.0) < 1.0 # d>0 -> sim<1
    assert model._distance_to_similarity(2.0) < model._distance_to_similarity(1.0) # Larger dist -> smaller sim

    # Other (Manhattan, Hamming - using fallback 1 / (1 + d))
    model.metric = 'manhattan'
    assert abs(model._distance_to_similarity(0.0) - 1.0) < eps
    assert abs(model._distance_to_similarity(1.0) - 0.5) < eps
    assert abs(model._distance_to_similarity(10.0) - (1.0/11.0)) < eps


@pytest.mark.skipif(not _model_exists, reason=f"Test model '{TEST_MODEL}' not found.")
def test_verbose_mode(default_params, capsys):
    """Test that verbose mode controls logging output."""
    # Test verbose=True
    params_verbose = default_params.copy()
    params_verbose["verbose"] = True
    model_verbose = SentenceTransformerContentBased(**params_verbose)
    model_verbose._fit(INTERACTION_MATRIX.copy())
    captured_verbose = capsys.readouterr()
    assert "Fitting SentenceTransformerContentBased" in captured_verbose.out
    assert "Batch encoding item content" in captured_verbose.out
    assert "Building Annoy index" in captured_verbose.out
    assert "Fit complete" in captured_verbose.out

    model_verbose._predict(INTERACTION_MATRIX.copy())
    captured_verbose_predict = capsys.readouterr()
    assert "Predicting recommendations" in captured_verbose_predict.out
    assert "Prediction complete" in captured_verbose_predict.out


    # Test verbose=False (default)
    params_quiet = default_params.copy()
    params_quiet["verbose"] = False
    model_quiet = SentenceTransformerContentBased(**params_quiet)
    model_quiet._fit(INTERACTION_MATRIX.copy())
    captured_quiet = capsys.readouterr()
    assert "Fitting SentenceTransformerContentBased" not in captured_quiet.out
    assert captured_quiet.out == "" # Expect no output

    model_quiet._predict(INTERACTION_MATRIX.copy())
    captured_quiet_predict = capsys.readouterr()
    assert "Predicting recommendations" not in captured_quiet_predict.out
    assert captured_quiet_predict.out == "" # Expect no output


# Recpack compatibility test (though not strictly necessary for unit testing the class logic)
def test_get_params(default_params):
     """Test the get_params method inherited from Algorithm."""
     model = SentenceTransformerContentBased(**default_params)
     params = model.get_params()
     # Remove sentencetransformer and annoy_index as they are not primitive types expected by get_params default
     # params.pop('sentencetransformer', None)
     # params.pop('annoy_index', None)
     # params.pop('_item_embeddings', None)
     # params.pop('_users_in_annoy', None)
     # params.pop('_user_offset', None)
     # params.pop('_embedding_dim', None) # Internal attribute

     # The base get_params() in sklearn/recpack typically returns constructor args
     expected_keys = default_params.keys()
     assert set(params.keys()) >= set(expected_keys) # Might include inferred params like _embedding_dim depending on base impl
     for key in expected_keys:
         assert params[key] == default_params[key]

# Optional: Test _eliminate_empty_users if it's critical
# This might require more setup specific to how recpack uses it.
# For now, focusing on the core logic of SentenceTransformerContentBased. 