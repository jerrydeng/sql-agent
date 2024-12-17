# test_sql_agent.py
import pytest
from sql_agent import SQLAgent

@pytest.fixture
def agent():
    """Create a test SQLAgent instance with SQLite sample database."""
    db_uri = "sqlite:///Chinook.db"
    agt = SQLAgent(db_uri)
    return agt

def test_basic_query(agent: SQLAgent):
    """Test a simple query about artists."""
    question = "How many artists are in the database?"
    result = agent.invoke(question)
    assert result is not None
    assert isinstance(result, str)
    # The Chinook database has 275 artists
    assert "275" in result.lower()

def test_complex_query(agent: SQLAgent):
    """Test a more complex query involving joins."""
    question = "What are the top 3 genres by number of tracks?"
    result = agent.invoke(question)
    assert result is not None
    assert isinstance(result, str)
    # Rock should be the most common genre in Chinook
    assert "rock" in result.lower()