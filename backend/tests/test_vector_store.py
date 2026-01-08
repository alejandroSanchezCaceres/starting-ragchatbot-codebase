"""Tests for VectorStore search behavior"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from vector_store import VectorStore, SearchResults


class TestSearchResults:
    """Test SearchResults dataclass methods"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'val1'}, {'key': 'val2'}]],
            'distances': [[0.1, 0.2]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key': 'val1'}, {'key': 'val2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        error_msg = "Database connection failed"
        results = SearchResults.empty(error_msg)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg
        assert results.is_empty()

    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty()

    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(
            documents=['doc1'],
            metadata=[{'key': 'val'}],
            distances=[0.1]
        )
        assert not results.is_empty()


class TestVectorStoreSearch:
    """Test VectorStore search method"""

    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('vector_store.chromadb.PersistentClient')
    def test_search_with_zero_max_results(self, mock_client_class, mock_embedding_func):
        """Test search behavior when MAX_RESULTS=0 (the bug scenario)"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_embedding_func.return_value = Mock()

        # Create VectorStore with max_results=0 (simulating the bug)
        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=0)

        # Mock the query to return empty when n_results=0
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        # Execute search
        results = store.search(query="test query")

        # Verify ChromaDB was called with n_results=0
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args
        assert call_args[1]['n_results'] == 0

        # Results should be empty
        assert results.is_empty()

    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('vector_store.chromadb.PersistentClient')
    def test_search_with_valid_max_results(self, mock_client_class, mock_embedding_func):
        """Test search behavior with MAX_RESULTS=5 (expected behavior)"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_embedding_func.return_value = Mock()

        # Create VectorStore with max_results=5
        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=5)

        # Mock the query to return results
        mock_collection.query.return_value = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course_title': 'MCP'}, {'course_title': 'MCP'}]],
            'distances': [[0.1, 0.2]]
        }

        # Execute search
        results = store.search(query="test query")

        # Verify ChromaDB was called with n_results=5
        call_args = mock_collection.query.call_args
        assert call_args[1]['n_results'] == 5

        # Results should have documents
        assert not results.is_empty()
        assert len(results.documents) == 2

    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('vector_store.chromadb.PersistentClient')
    def test_search_with_explicit_limit_overrides_max_results(self, mock_client_class, mock_embedding_func):
        """Test that explicit limit parameter overrides max_results"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_embedding_func.return_value = Mock()

        # Create VectorStore with max_results=5
        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=5)

        # Mock the query
        mock_collection.query.return_value = {
            'documents': [['doc1', 'doc2', 'doc3']],
            'metadatas': [[{}, {}, {}]],
            'distances': [[0.1, 0.2, 0.3]]
        }

        # Execute search with explicit limit=10
        results = store.search(query="test query", limit=10)

        # Verify ChromaDB was called with n_results=10 (not 5)
        call_args = mock_collection.query.call_args
        assert call_args[1]['n_results'] == 10

    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('vector_store.chromadb.PersistentClient')
    def test_search_exception_returns_error_result(self, mock_client_class, mock_embedding_func):
        """Test that search exceptions are caught and returned as error"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_embedding_func.return_value = Mock()

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=5)

        # Mock the query to raise exception
        mock_collection.query.side_effect = Exception("ChromaDB connection failed")

        # Execute search
        results = store.search(query="test query")

        # Should return error result
        assert results.error is not None
        assert "Search error" in results.error
        assert "ChromaDB connection failed" in results.error
        assert results.is_empty()


class TestVectorStoreCourseResolution:
    """Test course name resolution functionality"""

    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('vector_store.chromadb.PersistentClient')
    def test_resolve_course_name_success(self, mock_client_class, mock_embedding_func):
        """Test successful course name resolution"""
        # Setup mocks
        mock_client = Mock()
        mock_catalog = Mock()
        mock_content = Mock()
        mock_embedding_func.return_value = Mock()

        # Return different collections for catalog and content
        def get_collection_side_effect(name, embedding_function):
            if name == "course_catalog":
                return mock_catalog
            elif name == "course_content":
                return mock_content
            return Mock()

        mock_client.get_or_create_collection.side_effect = get_collection_side_effect
        mock_client_class.return_value = mock_client

        # Mock catalog query to return resolved title
        mock_catalog.query.return_value = {
            'documents': [['Introduction to MCP']],
            'metadatas': [[{'title': 'Introduction to MCP'}]]
        }

        # Mock content query
        mock_content.query.return_value = {
            'documents': [['content1']],
            'metadatas': [[{'course_title': 'Introduction to MCP'}]],
            'distances': [[0.1]]
        }

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=5)

        # Search with partial course name
        results = store.search(query="what is MCP?", course_name="MCP")

        # Verify catalog was queried for course name
        mock_catalog.query.assert_called_once()
        catalog_call = mock_catalog.query.call_args
        assert catalog_call[1]['query_texts'] == ['MCP']

        # Verify content was queried with resolved title as filter
        mock_content.query.assert_called_once()
        content_call = mock_content.query.call_args
        assert content_call[1]['where'] == {'course_title': 'Introduction to MCP'}

    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('vector_store.chromadb.PersistentClient')
    def test_resolve_course_name_not_found(self, mock_client_class, mock_embedding_func):
        """Test course name resolution when course not found"""
        # Setup mocks
        mock_client = Mock()
        mock_catalog = Mock()
        mock_content = Mock()
        mock_embedding_func.return_value = Mock()

        def get_collection_side_effect(name, embedding_function):
            if name == "course_catalog":
                return mock_catalog
            elif name == "course_content":
                return mock_content
            return Mock()

        mock_client.get_or_create_collection.side_effect = get_collection_side_effect
        mock_client_class.return_value = mock_client

        # Mock catalog query to return empty results
        mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=5)

        # Search with non-existent course name
        results = store.search(query="what is MCP?", course_name="NonexistentCourse")

        # Should return error result
        assert results.error is not None
        assert "No course found matching 'NonexistentCourse'" in results.error
        assert results.is_empty()

        # Content should NOT be queried if course name resolution failed
        mock_content.query.assert_not_called()


class TestVectorStoreFilterBuilding:
    """Test filter building for various parameter combinations"""

    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('vector_store.chromadb.PersistentClient')
    def test_build_filter_no_parameters(self, mock_client_class, mock_embedding_func):
        """Test filter with no course or lesson specified"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_embedding_func.return_value = Mock()

        # Mock query
        mock_collection.query.return_value = {
            'documents': [['doc1']],
            'metadatas': [[{}]],
            'distances': [[0.1]]
        }

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=5)

        # Search without filters
        store.search(query="test query")

        # Verify where clause is None
        call_args = mock_collection.query.call_args
        assert call_args[1]['where'] is None

    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('vector_store.chromadb.PersistentClient')
    def test_build_filter_lesson_only(self, mock_client_class, mock_embedding_func):
        """Test filter with only lesson_number specified"""
        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_embedding_func.return_value = Mock()

        # Mock query
        mock_collection.query.return_value = {
            'documents': [['doc1']],
            'metadatas': [[{}]],
            'distances': [[0.1]]
        }

        # Create VectorStore
        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=5)

        # Search with lesson filter only
        store.search(query="test query", lesson_number=2)

        # Verify where clause has lesson_number
        call_args = mock_collection.query.call_args
        assert call_args[1]['where'] == {'lesson_number': 2}
