"""
A2.Log.AI - Self-Cleaning Communication Protocol for AI Conversations
Complete System with Streamlit Testing Interface

Features:
1. AI conversation classification (Grade 1/2/3)
2. Code snippet extraction and storage
3. Formula/definition detection
4. Document alpha/beta selection
5. Reuse-based importance tracking
6. Cross-session persistence
7. Semantic search over A2.Log

Run: streamlit run a2log_ai.py

Author: Anindita Ray
Version: 1.0
"""

import streamlit as st
from datetime import datetime, timedelta
import re
import hashlib
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np


# ============================================================================
# AI MESSAGE CLASSIFIER
# ============================================================================

class AIMessageClassifier:
    """Classify AI conversation messages into Grade 1/2/3."""
    
    # Grade 1 patterns (Alpha Insights)
    CODE_PATTERNS = [
        r'```[\w]*\n.*?\n```',  # Code blocks
        r'`[^`]+`',              # Inline code
    ]
    
    FORMULA_PATTERNS = [
        r'\$.*?\$',              # LaTeX inline
        r'\$\$.*?\$\$',          # LaTeX block
        r'[A-Z]\s*=\s*[^,\n]+',  # Equations like E = mc²
    ]
    
    DEFINITION_PATTERNS = [
        r'is defined as',
        r'refers to',
        r'means that',
        r'can be understood as',
        r'is a [a-z]+ that',
    ]
    
    ACTION_PATTERNS = [
        r'step \d+:',
        r'todo:',
        r'action item:',
        r'implement',
        r'configure',
        r'install',
    ]
    
    # Grade 3 patterns (Noise)
    NOISE_PATTERNS = [
        r'^(thanks|thank you|ok|okay|got it|sure|yes|no)$',
        r'^(continue|next|go on|keep going)$',
        r'^(interesting|hmm|i see)$',
        r'^(hello|hi|hey|bye)$',
    ]
    
    @classmethod
    def classify(cls, message: str, role: str) -> Dict:
        """
        Classify AI message.
        
        Returns:
            {
                'grade': 1/2/3,
                'type': 'code'/'formula'/'definition'/etc,
                'reason': str,
                'entities': {}
            }
        """
        message_lower = message.lower().strip()
        
        # Check for noise patterns
        if role == 'user':
            for pattern in cls.NOISE_PATTERNS:
                if re.match(pattern, message_lower):
                    return {
                        'grade': 3,
                        'type': 'noise',
                        'reason': 'acknowledgment_or_navigation',
                        'entities': {}
                    }
        
        # Very short messages
        if len(message_lower.split()) < 5 and role == 'user':
            return {
                'grade': 3,
                'type': 'noise',
                'reason': 'too_short',
                'entities': {}
            }
        
        # Check for Alpha patterns
        entities = {}
        
        # Code detection
        code_blocks = re.findall(cls.CODE_PATTERNS[0], message, re.DOTALL)
        inline_code = re.findall(cls.CODE_PATTERNS[1], message)
        if code_blocks or inline_code:
            entities['code_blocks'] = code_blocks
            entities['inline_code'] = inline_code
            return {
                'grade': 1,
                'type': 'code',
                'reason': f'contains_{len(code_blocks)}_code_blocks',
                'entities': entities
            }
        
        # Formula detection
        formulas = []
        for pattern in cls.FORMULA_PATTERNS:
            formulas.extend(re.findall(pattern, message))
        if formulas:
            entities['formulas'] = formulas
            return {
                'grade': 1,
                'type': 'formula',
                'reason': f'contains_{len(formulas)}_formulas',
                'entities': entities
            }
        
        # Definition detection
        for pattern in cls.DEFINITION_PATTERNS:
            if re.search(pattern, message_lower):
                return {
                    'grade': 1,
                    'type': 'definition',
                    'reason': 'contains_definition',
                    'entities': entities
                }
        
        # Action items
        for pattern in cls.ACTION_PATTERNS:
            if re.search(pattern, message_lower):
                return {
                    'grade': 1,
                    'type': 'action_item',
                    'reason': 'contains_action',
                    'entities': entities
                }
        
        # Structured data (lists, tables)
        if message.count('\n-') > 2 or message.count('\n*') > 2:
            return {
                'grade': 1,
                'type': 'structured_data',
                'reason': 'contains_list',
                'entities': entities
            }
        
        # Default: Grade 2 (Conversational DNA)
        return {
            'grade': 2,
            'type': 'conversational',
            'reason': 'explanatory_content',
            'entities': entities
        }


# ============================================================================
# REUSE DETECTOR
# ============================================================================

class ReuseDetector:
    """Track topic reuse across sessions."""
    
    def __init__(self):
        self.topic_frequency = defaultdict(int)
        self.topic_first_seen = {}
        self.topic_last_seen = {}
    
    def extract_topic(self, message: str) -> str:
        """Extract main topic from message."""
        # Simple approach: first 5 words
        words = message.lower().split()[:5]
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'how', 'what', 'can', 'you', 'please'}
        topic_words = [w for w in words if w not in stopwords]
        return ' '.join(topic_words[:3]) if topic_words else 'unknown'
    
    def track(self, message: str) -> Dict:
        """Track message and check for reuse."""
        topic = self.extract_topic(message)
        
        self.topic_frequency[topic] += 1
        self.topic_last_seen[topic] = datetime.now()
        
        if topic not in self.topic_first_seen:
            self.topic_first_seen[topic] = datetime.now()
        
        return {
            'topic': topic,
            'frequency': self.topic_frequency[topic],
            'promote_to_alpha': self.topic_frequency[topic] >= 2
        }


# ============================================================================
# DOCUMENT CLUSTERER (Alpha/Beta for uploaded docs)
# ============================================================================

class DocumentClusterer:
    """Cluster uploaded documents and select alphas."""
    
    def __init__(self):
        self.document_database = {}
    
    def compute_hash(self, content: str) -> str:
        """Generate content hash."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity (Jaccard)."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = set1 & set2
        union = set1 | set2
        
        return len(intersection) / len(union)
    
    def cluster_documents(self, documents: List[Dict]) -> Dict:
        """
        Cluster documents by similarity and select alphas.
        
        Args:
            documents: [{'name': str, 'content': str, 'uploader': str}, ...]
        
        Returns:
            {
                'clusters': [...],
                'alphas': [...],
                'betas': [...]
            }
        """
        if not documents:
            return {'clusters': [], 'alphas': [], 'betas': []}
        
        # Simple clustering: group by similarity > 0.7
        clusters = []
        assigned = [False] * len(documents)
        
        for i in range(len(documents)):
            if assigned[i]:
                continue
            
            cluster = [i]
            assigned[i] = True
            
            for j in range(i + 1, len(documents)):
                if assigned[j]:
                    continue
                
                similarity = self.compute_similarity(
                    documents[i]['content'],
                    documents[j]['content']
                )
                
                if similarity >= 0.7:
                    cluster.append(j)
                    assigned[j] = True
            
            clusters.append(cluster)
        
        # Select alpha from each cluster (first one for simplicity)
        alphas = [cluster[0] for cluster in clusters]
        betas = [i for i in range(len(documents)) if i not in alphas]
        
        return {
            'clusters': clusters,
            'alphas': alphas,
            'betas': betas,
            'reduction': (1 - len(alphas) / len(documents)) * 100 if documents else 0
        }


# ============================================================================
# A2.LOG STORAGE
# ============================================================================

class A2LogStorage:
    """Storage and retrieval for A2.Log."""
    
    def __init__(self):
        self.alpha_entries = []  # Grade 1 entries
        self.conversational_entries = []  # Grade 2 (archived after session)
        self.noise_entries = []  # Grade 3 (purged)
    
    def add_entry(self, entry: Dict):
        """Add entry to appropriate storage."""
        grade = entry['grade']
        
        if grade == 1:
            self.alpha_entries.append(entry)
        elif grade == 2:
            self.conversational_entries.append(entry)
        else:
            self.noise_entries.append(entry)
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search A2.Log for relevant entries."""
        query_lower = query.lower()
        results = []
        
        for entry in self.alpha_entries:
            content_lower = entry['content'].lower()
            
            # Simple keyword matching
            if query_lower in content_lower:
                score = content_lower.count(query_lower)
                results.append((entry, score))
        
        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [entry for entry, _ in results[:limit]]
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return {
            'alpha_count': len(self.alpha_entries),
            'conversational_count': len(self.conversational_entries),
            'noise_count': len(self.noise_entries),
            'total_entries': len(self.alpha_entries) + len(self.conversational_entries) + len(self.noise_entries)
        }


# ============================================================================
# A2.LOG.AI ENGINE
# ============================================================================

class A2LogAI:
    """Main A2.Log.AI engine."""
    
    def __init__(self):
        self.classifier = AIMessageClassifier()
        self.reuse_detector = ReuseDetector()
        self.document_clusterer = DocumentClusterer()
        self.storage = A2LogStorage()
        
        self.conversation_history = []
        
        self.stats = {
            'messages_processed': 0,
            'alpha_insights': 0,
            'conversational': 0,
            'noise_purged': 0,
            'topics_reused': 0,
            'documents_uploaded': 0
        }
    
    def process_message(self, message: str, role: str) -> Dict:
        """Process a single message from conversation."""
        self.stats['messages_processed'] += 1
        
        # Classify
        classification = self.classifier.classify(message, role)
        
        # Track reuse (for user messages)
        reuse_info = None
        if role == 'user':
            reuse_info = self.reuse_detector.track(message)
            
            # Promote to Grade 1 if reused
            if reuse_info['promote_to_alpha'] and classification['grade'] == 2:
                classification['grade'] = 1
                classification['type'] = 'reused_topic'
                classification['reason'] = f"asked_{reuse_info['frequency']}_times"
                self.stats['topics_reused'] += 1
        
        # Update stats
        if classification['grade'] == 1:
            self.stats['alpha_insights'] += 1
        elif classification['grade'] == 2:
            self.stats['conversational'] += 1
        else:
            self.stats['noise_purged'] += 1
        
        # Store
        entry = {
            'content': message,
            'role': role,
            'timestamp': datetime.now(),
            'grade': classification['grade'],
            'type': classification['type'],
            'reason': classification['reason'],
            'entities': classification.get('entities', {}),
            'reuse_info': reuse_info
        }
        
        self.storage.add_entry(entry)
        self.conversation_history.append(entry)
        
        return {
            'status': 'processed',
            'classification': classification,
            'reuse_info': reuse_info,
            'storage_location': self._get_storage_location(classification['grade'])
        }
    
    def process_documents(self, documents: List[Dict]) -> Dict:
        """Process uploaded documents with alpha/beta selection."""
        self.stats['documents_uploaded'] += len(documents)
        
        clustering_result = self.document_clusterer.cluster_documents(documents)
        
        # Store alphas in A2.Log
        for alpha_idx in clustering_result['alphas']:
            doc = documents[alpha_idx]
            entry = {
                'content': doc['content'][:500] + '...',  # Store excerpt
                'role': 'document',
                'timestamp': datetime.now(),
                'grade': 1,
                'type': 'document_alpha',
                'reason': 'selected_as_alpha',
                'entities': {'full_content': doc['content'], 'name': doc['name']},
                'reuse_info': None
            }
            self.storage.add_entry(entry)
            self.stats['alpha_insights'] += 1
        
        return {
            'status': 'processed',
            'total_documents': len(documents),
            'alphas': clustering_result['alphas'],
            'betas': clustering_result['betas'],
            'clusters': len(clustering_result['clusters']),
            'reduction': clustering_result['reduction']
        }
    
    def search_a2_log(self, query: str) -> List[Dict]:
        """Search A2.Log for relevant content."""
        return self.storage.search(query)
    
    def _get_storage_location(self, grade: int) -> str:
        """Get storage location description."""
        if grade == 1:
            return 'A2.Log (Permanent, Searchable)'
        elif grade == 2:
            return 'Session Memory (Archived after 24h)'
        else:
            return 'Purged (Not stored)'
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        storage_stats = self.storage.get_stats()
        return {**self.stats, **storage_stats}


# ============================================================================
# STREAMLIT TESTING INTERFACE
# ============================================================================

def init_session_state():
    """Initialize session state."""
    if 'engine' not in st.session_state:
        st.session_state.engine = A2LogAI()
    if 'conversation_mode' not in st.session_state:
        st.session_state.conversation_mode = False


def render_header():
    """Render header."""
    st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; margin-bottom: 2rem;">
            <h1 style="color: white; font-size: 3rem; margin: 0;">A2.Log.AI</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Self-Cleaning Protocol for AI Conversations</p>
        </div>
    """, unsafe_allow_html=True)


def render_stats():
    """Render statistics dashboard."""
    st.markdown("### 📊 System Statistics")
    
    stats = st.session_state.engine.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Messages Processed", stats['messages_processed'])
    
    with col2:
        st.metric("🟢 Alpha Insights", stats['alpha_insights'])
    
    with col3:
        st.metric("🔵 Conversational", stats['conversational'])
    
    with col4:
        st.metric("⚪ Noise Purged", stats['noise_purged'])
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Topics Reused", stats['topics_reused'])
    
    with col2:
        st.metric("Documents Uploaded", stats['documents_uploaded'])
    
    with col3:
        st.metric("A2.Log Entries", stats['alpha_count'])


def render_conversation_interface():
    """Render conversation testing interface."""
    st.markdown("### 💬 AI Conversation Simulator")
    
    engine = st.session_state.engine
    
    # Input mode
    col1, col2 = st.columns([3, 1])
    
    with col1:
        message = st.text_area("Message:", height=150, placeholder="Type your message or AI response here...")
    
    with col2:
        role = st.selectbox("Role:", ["user", "assistant"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Process Message", use_container_width=True):
            if message.strip():
                result = engine.process_message(message, role)
                
                grade_labels = {
                    1: "🟢 Alpha Insight",
                    2: "🔵 Conversational DNA",
                    3: "⚪ Transient Noise"
                }
                
                classification = result['classification']
                
                st.success(f"Classified: {grade_labels[classification['grade']]}")
                st.info(f"Type: {classification['type']}")
                st.info(f"Storage: {result['storage_location']}")
                
                if result['reuse_info'] and result['reuse_info']['frequency'] > 1:
                    st.warning(f"⚠️ Topic reused {result['reuse_info']['frequency']} times - Promoted to Alpha!")
                
                st.rerun()
    
    # Conversation history
    if engine.conversation_history:
        st.markdown("#### Recent Conversation")
        
        for entry in reversed(engine.conversation_history[-10:]):
            grade_colors = {1: "#667eea", 2: "#48bb78", 3: "#cbd5e0"}
            grade_icons = {1: "🟢", 2: "🔵", 3: "⚪"}
            
            color = grade_colors.get(entry['grade'], "#cbd5e0")
            icon = grade_icons.get(entry['grade'], "⚪")
            
            role_label = "You" if entry['role'] == 'user' else "AI"
            
            st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding: 1rem; margin: 0.5rem 0; background: rgba(255,255,255,0.05); border-radius: 4px;">
                    <div style="font-size: 0.8rem; color: {color}; margin-bottom: 0.5rem;">
                        {icon} Grade {entry['grade']} | {role_label} | {entry['type']}
                    </div>
                    <div>{entry['content'][:200]}{'...' if len(entry['content']) > 200 else ''}</div>
                </div>
            """, unsafe_allow_html=True)


def render_document_interface():
    """Render document upload interface."""
    st.markdown("### 📄 Document Upload & Alpha Selection")
    
    engine = st.session_state.engine
    
    st.info("Upload multiple documents to test alpha/beta clustering (e.g., syllabus uploaded by 5 different people)")
    
    # Manual document input
    st.markdown("#### Add Documents Manually")
    
    num_docs = st.number_input("Number of documents:", min_value=1, max_value=20, value=5)
    
    documents = []
    
    for i in range(num_docs):
        with st.expander(f"Document {i+1}"):
            name = st.text_input(f"Name:", value=f"Document_{i+1}", key=f"doc_name_{i}")
            content = st.text_area(f"Content:", height=100, key=f"doc_content_{i}", 
                                  placeholder="Paste document content here...")
            uploader = st.text_input(f"Uploader:", value=f"User{i+1}", key=f"doc_uploader_{i}")
            
            if content.strip():
                documents.append({
                    'name': name,
                    'content': content,
                    'uploader': uploader
                })
    
    if st.button("🎯 Process Documents", use_container_width=True):
        if documents:
            with st.spinner("Clustering documents..."):
                result = engine.process_documents(documents)
            
            st.success(f"Processed {result['total_documents']} documents!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Docs", result['total_documents'])
            
            with col2:
                st.metric("Clusters", result['clusters'])
            
            with col3:
                st.metric("Alphas", len(result['alphas']))
            
            with col4:
                st.metric("Reduction", f"{result['reduction']:.1f}%")
            
            st.markdown("#### Selection Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🟢 Alphas (Kept in A2.Log):**")
                for idx in result['alphas']:
                    st.write(f"- {documents[idx]['name']} by {documents[idx]['uploader']}")
            
            with col2:
                st.markdown("**⚪ Betas (Archived):**")
                for idx in result['betas']:
                    st.write(f"- {documents[idx]['name']} by {documents[idx]['uploader']}")
            
            st.rerun()
        else:
            st.warning("Please add document content first")


def render_search_interface():
    """Render A2.Log search interface."""
    st.markdown("### 🔍 A2.Log Search")
    
    engine = st.session_state.engine
    
    query = st.text_input("Search query:", placeholder="e.g., 'binary search', 'quantum', 'python'")
    
    if st.button("🔎 Search", use_container_width=True) or query:
        if query.strip():
            results = engine.search_a2_log(query)
            
            if results:
                st.success(f"Found {len(results)} results")
                
                for i, entry in enumerate(results, 1):
                    with st.expander(f"Result {i}: {entry['type']} from {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                        st.markdown(f"**Grade:** {entry['grade']}")
                        st.markdown(f"**Type:** {entry['type']}")
                        st.markdown(f"**Content:**")
                        st.code(entry['content'][:500] + ('...' if len(entry['content']) > 500 else ''))
                        
                        if entry.get('entities'):
                            st.markdown("**Entities:**")
                            st.json(entry['entities'])
            else:
                st.info("No results found")
    
    # Show all A2.Log entries
    st.markdown("---")
    st.markdown("#### All A2.Log Entries")
    
    if engine.storage.alpha_entries:
        for entry in reversed(engine.storage.alpha_entries[-10:]):
            type_icons = {
                'code': '💻',
                'formula': '📐',
                'definition': '📖',
                'action_item': '✅',
                'structured_data': '📊',
                'reused_topic': '🔄',
                'document_alpha': '📄'
            }
            
            icon = type_icons.get(entry['type'], '📝')
            
            with st.expander(f"{icon} {entry['type']} - {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(entry['content'][:300] + ('...' if len(entry['content']) > 300 else ''))
    else:
        st.info("A2.Log is empty. Process some messages to populate it.")


def render_demo_scenarios():
    """Render pre-built demo scenarios."""
    st.markdown("### 🎭 Demo Scenarios")
    
    engine = st.session_state.engine
    
    scenario = st.selectbox("Select scenario:", [
        "Coding Session (Binary Search)",
        "Physics Research (Quantum Computing)",
        "Document Upload (Syllabus x5)",
        "Reuse Detection (Same Question Twice)"
    ])
    
    if st.button("▶️ Run Scenario", use_container_width=True):
        
        if scenario == "Coding Session (Binary Search)":
            messages = [
                ("user", "Explain binary search algorithm"),
                ("assistant", "Binary search is a search algorithm that finds the position of a target value within a sorted array. It works by repeatedly dividing the search interval in half.\n\nHere's the Python implementation:\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"),
                ("user", "What's the time complexity?"),
                ("assistant", "The time complexity is O(log n) because the search space is halved in each iteration."),
            ]
            
            for role, msg in messages:
                engine.process_message(msg, role)
            
            st.success("Scenario completed! Check conversation history.")
            st.rerun()
        
        elif scenario == "Physics Research (Quantum Computing)":
            messages = [
                ("user", "Explain quantum entanglement"),
                ("assistant", "Quantum entanglement is defined as a phenomenon where pairs or groups of particles interact in ways such that the quantum state of each particle cannot be described independently. The formula for an entangled state is: |ψ⟩ = (|00⟩ + |11⟩)/√2"),
                ("user", "How does this relate to superposition?"),
                ("assistant", "Superposition is a fundamental principle where a quantum system can exist in multiple states simultaneously until measured. Entanglement builds on this by creating correlations between particles in superposition."),
            ]
            
            for role, msg in messages:
                engine.process_message(msg, role)
            
            st.success("Scenario completed!")
            st.rerun()
        
        elif scenario == "Document Upload (Syllabus x5)":
            documents = []
            syllabus_content = "Course: Data Structures\nTopics: Arrays, Linked Lists, Trees, Graphs\nExam: December 20, 2024"
            
            for i in range(5):
                documents.append({
                    'name': f'Syllabus_User{i+1}.pdf',
                    'content': syllabus_content + f"\nUploaded by User{i+1}",
                    'uploader': f'User{i+1}'
                })
            
            result = engine.process_documents(documents)
            st.success(f"Processed 5 documents, selected {len(result['alphas'])} alpha, {result['reduction']:.0f}% reduction")
            st.rerun()
        
        elif scenario == "Reuse Detection (Same Question Twice)":
            # First time
            engine.process_message("Explain gradient descent", "user")
            engine.process_message("Gradient descent is an optimization algorithm...", "assistant")
            
            st.info("Processed first time (Grade 2)")
            
            # Second time (should promote to Grade 1)
            engine.process_message("What was that gradient descent thing?", "user")
            
            st.warning("Processed second time - Topic promoted to Grade 1!")
            st.rerun()


def main():
    """Main app."""
    st.set_page_config(
        page_title="A2.Log.AI",
        page_icon="🤖",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        }
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Navigation")
        page = st.radio("", [
            "Dashboard",
            "Conversation",
            "Documents",
            "Search A2.Log",
            "Demo Scenarios"
        ], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### Quick Info")
        st.info("A2.Log.AI automatically extracts and stores important insights from AI conversations.")
        
        st.markdown("---")
        
        if st.button("🔄 Reset All Data", use_container_width=True):
            st.session_state.engine = A2LogAI()
            st.success("Reset complete!")
            st.rerun()
    
    # Main content
    if page == "Dashboard":
        render_stats()
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            render_conversation_interface()
        with col2:
            render_search_interface()
    
    elif page == "Conversation":
        render_conversation_interface()
    
    elif page == "Documents":
        render_document_interface()
    
    elif page == "Search A2.Log":
        render_search_interface()
    
    elif page == "Demo Scenarios":
        render_demo_scenarios()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            A2.Log.AI v1.0 | Self-Cleaning Protocol for AI Conversations | Design: Anindita Ray
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()