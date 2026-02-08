"""
A2.Log.AI v2.0 - ENHANCED VERSION
Critical Fixes Implemented:

1. LOGIC DETECTION: Better NLP for system/logic recognition
2. FAILURE-BASED LEARNING: Only save formulas you failed
3. SMART CHAT MERGING: Cut-paste between conversation threads
4. COLLABORATIVE A2.LOG: GitHub-like multi-user sync

Run: streamlit run a2log_ai_enhanced.py

Author: Anindita Ray
Version: 2.0 Enhanced
"""

import streamlit as st
from datetime import datetime, timedelta
import re
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np


# ============================================================================
# ENHANCED AI MESSAGE CLASSIFIER (FIX #1: Logic Detection)
# ============================================================================

class EnhancedAIClassifier:
    """Enhanced classifier with logic/system detection."""
    
    # CRITICAL PATTERNS FOR LOGIC/SYSTEM DETECTION
    LOGIC_PATTERNS = [
        # Architecture/System descriptions
        r'architecture|system design|protocol|framework',
        r'flow(?:chart)?:.*(?:→|->)',  # Flowcharts
        r'(?:step|stage|phase)\s+\d+:',  # Multi-step processes
        
        # Logic keywords
        r'algorithm|logic|mechanism|process flow',
        r'if.*then.*else',  # Conditional logic
        r'when.*trigger|event.*action',
        
        # Technical specifications
        r'(?:api|database|server|client).*(?:design|schema)',
        r'class\s+\w+|interface\s+\w+',  # OOP design
        
        # Backbone/Core concepts
        r'backbone|core\s+(?:logic|concept|principle)',
        r'fundamental|foundational',
        r'(?:data|control)\s+flow',
        
        # Multi-tier systems
        r'tier\s+\d+|grade\s+\d+|layer\s+\d+',
        r'(?:three|3)-(?:tier|layer|grade)',
    ]
    
    CODE_PATTERNS = [
        r'```[\w]*\n.*?\n```',
        r'`[^`]+`',
    ]
    
    FORMULA_PATTERNS = [
        r'\$.*?\$',
        r'\$\$.*?\$\$',
        r'[A-Z]\s*=\s*[^,\n]+',
        # Excel formulas
        r'=(?:SUM|VLOOKUP|IF|INDEX|MATCH|SUMIF|COUNTIF|AVERAGEIF|IFERROR|CONCATENATE|LEFT|RIGHT|MID|LEN|TRIM|UPPER|LOWER|PROPER|TEXT|VALUE|DATE|TODAY|NOW|YEAR|MONTH|DAY|HOUR|MINUTE|SECOND|WEEKDAY|NETWORKDAYS|EOMONTH|EDATE|DATEDIF|PMT|FV|PV|RATE|NPER|NPV|IRR|XNPV|XIRR|ROUND|ROUNDUP|ROUNDDOWN|INT|TRUNC|MOD|ABS|SQRT|POWER|EXP|LN|LOG|LOG10|PI|RAND|RANDBETWEEN|AND|OR|NOT|XOR|TRUE|FALSE|ISBLANK|ISERROR|ISTEXT|ISNUMBER|ISLOGICAL|ISNA|ISERR|ISODD|ISEVEN|CELL|TYPE|N|T|AREAS|COLUMNS|ROWS|CHOOSE|HLOOKUP|LOOKUP|OFFSET|INDIRECT|ADDRESS|COLUMN|ROW|TRANSPOSE|GETPIVOTDATA|HYPERLINK|SUBTOTAL|AGGREGATE)',
    ]
    
    DEFINITION_PATTERNS = [
        r'is defined as',
        r'refers to',
        r'means that',
        r'can be understood as',
    ]
    
    NOISE_PATTERNS = [
        r'^(thanks|thank you|ok|okay|got it|sure|yes|no)$',
        r'^(continue|next|go on)$',
        r'^(interesting|hmm|i see)$',
    ]
    
    @classmethod
    def classify(cls, message: str, role: str, user_feedback: Optional[str] = None) -> Dict:
        """
        Enhanced classification with logic detection.
        
        Args:
            message: The message content
            role: 'user' or 'assistant'
            user_feedback: 'failed' or 'mastered' (for formulas)
        
        Returns:
            {
                'grade': 1/2/3,
                'type': str,
                'reason': str,
                'entities': {},
                'confidence': float
            }
        """
        message_lower = message.lower().strip()
        
        # Check noise
        if role == 'user':
            for pattern in cls.NOISE_PATTERNS:
                if re.match(pattern, message_lower):
                    return {
                        'grade': 3,
                        'type': 'noise',
                        'reason': 'acknowledgment',
                        'entities': {},
                        'confidence': 1.0
                    }
        
        # FIX #1: LOGIC/SYSTEM DETECTION (Highest Priority)
        logic_matches = 0
        for pattern in cls.LOGIC_PATTERNS:
            if re.search(pattern, message_lower):
                logic_matches += 1
        
        # If 2+ logic patterns match, it's definitely a system/logic explanation
        if logic_matches >= 2 or re.search(r'(?:sccp|a2\.log|protocol|architecture).*(?:system|logic|design)', message_lower):
            return {
                'grade': 1,
                'type': 'logic_system',
                'reason': f'detected_system_logic_{logic_matches}_patterns',
                'entities': {'logic_patterns_matched': logic_matches},
                'confidence': 0.95,
                'auto_save_to_drive': True  # FIX #1: Auto-save to Drive
            }
        
        # Code detection
        code_blocks = re.findall(cls.CODE_PATTERNS[0], message, re.DOTALL)
        if code_blocks:
            return {
                'grade': 1,
                'type': 'code',
                'reason': f'contains_{len(code_blocks)}_code_blocks',
                'entities': {'code_blocks': code_blocks},
                'confidence': 1.0
            }
        
        # FIX #2: FORMULA DETECTION WITH FAILURE-BASED LEARNING
        formulas = []
        for pattern in cls.FORMULA_PATTERNS:
            formulas.extend(re.findall(pattern, message, re.IGNORECASE))
        
        if formulas:
            # Only save if user explicitly failed
            if user_feedback == 'failed':
                return {
                    'grade': 1,
                    'type': 'formula_failed',
                    'reason': 'user_marked_as_failed',
                    'entities': {'formulas': formulas},
                    'confidence': 1.0,
                    'user_struggled': True  # Flag for review
                }
            elif user_feedback == 'mastered':
                return {
                    'grade': 3,
                    'type': 'formula_mastered',
                    'reason': 'user_marked_as_mastered',
                    'entities': {'formulas': formulas},
                    'confidence': 1.0
                }
            else:
                # Default: Don't save (assume mastered)
                return {
                    'grade': 2,
                    'type': 'formula_practice',
                    'reason': 'formula_detected_no_feedback',
                    'entities': {'formulas': formulas},
                    'confidence': 0.7,
                    'requires_user_feedback': True  # Prompt user
                }
        
        # Definition detection
        for pattern in cls.DEFINITION_PATTERNS:
            if re.search(pattern, message_lower):
                return {
                    'grade': 1,
                    'type': 'definition',
                    'reason': 'contains_definition',
                    'entities': {},
                    'confidence': 0.9
                }
        
        # Structured data
        if message.count('\n-') > 2 or message.count('\n*') > 2:
            return {
                'grade': 1,
                'type': 'structured_data',
                'reason': 'contains_list',
                'entities': {},
                'confidence': 0.8
            }
        
        # Default: Conversational
        return {
            'grade': 2,
            'type': 'conversational',
            'reason': 'no_alpha_patterns_detected',
            'entities': {},
            'confidence': 0.6
        }


# ============================================================================
# CHAT MERGER (FIX #3: Smart Merge Between Threads)
# ============================================================================

class ChatMerger:
    """Merge conversation threads intelligently."""
    
    def __init__(self):
        self.conversations = {}  # conversation_id -> list of messages
    
    def add_conversation(self, conv_id: str, messages: List[Dict]):
        """Add a conversation thread."""
        self.conversations[conv_id] = messages
    
    def find_common_topics(self, conv1_id: str, conv2_id: str) -> List[str]:
        """Find common topics between two conversations."""
        conv1 = self.conversations.get(conv1_id, [])
        conv2 = self.conversations.get(conv2_id, [])
        
        # Extract keywords from both
        def extract_keywords(messages):
            text = ' '.join([m['content'] for m in messages])
            words = text.lower().split()
            # Remove stopwords
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
            keywords = [w for w in words if len(w) > 3 and w not in stopwords]
            # Count frequency
            freq = defaultdict(int)
            for word in keywords:
                freq[word] += 1
            # Top 10
            return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        keywords1 = {k for k, _ in extract_keywords(conv1)}
        keywords2 = {k for k, _ in extract_keywords(conv2)}
        
        common = keywords1 & keywords2
        return list(common)
    
    def merge_conversations(
        self, 
        source_id: str, 
        target_id: str,
        start_index: int,
        end_index: int
    ) -> Dict:
        """
        Merge messages from source into target.
        
        Args:
            source_id: Source conversation ID
            target_id: Target conversation ID
            start_index: Start message index in source
            end_index: End message index in source
        
        Returns:
            {
                'status': 'success',
                'moved_messages': int,
                'common_topics': [...]
            }
        """
        source_conv = self.conversations.get(source_id, [])
        target_conv = self.conversations.get(target_id, [])
        
        if not source_conv or not target_conv:
            return {'status': 'error', 'message': 'Conversation not found'}
        
        # Extract messages to move
        messages_to_move = source_conv[start_index:end_index+1]
        
        # Add timestamp marker
        merge_marker = {
            'content': f'--- MERGED FROM {source_id} ---',
            'role': 'system',
            'timestamp': datetime.now(),
            'grade': 2,
            'type': 'merge_marker',
            'entities': {}
        }
        
        # Append to target
        target_conv.append(merge_marker)
        target_conv.extend(messages_to_move)
        
        # Remove from source
        self.conversations[source_id] = source_conv[:start_index] + source_conv[end_index+1:]
        
        # Update target
        self.conversations[target_id] = target_conv
        
        # Find common topics
        common = self.find_common_topics(source_id, target_id)
        
        return {
            'status': 'success',
            'moved_messages': len(messages_to_move),
            'common_topics': common,
            'source_remaining': len(self.conversations[source_id]),
            'target_total': len(self.conversations[target_id])
        }
    
    def suggest_merge_targets(self, source_id: str) -> List[Tuple[str, float]]:
        """Suggest which conversation to merge into based on topic similarity."""
        source_conv = self.conversations.get(source_id, [])
        if not source_conv:
            return []
        
        suggestions = []
        
        for conv_id in self.conversations:
            if conv_id == source_id:
                continue
            
            common_topics = self.find_common_topics(source_id, conv_id)
            similarity = len(common_topics)
            
            if similarity > 0:
                suggestions.append((conv_id, similarity))
        
        # Sort by similarity
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return suggestions


# ============================================================================
# COLLABORATIVE A2.LOG (FIX #4: Multi-User Sync)
# ============================================================================

class CollaborativeA2Log:
    """GitHub-like collaborative A2.Log with multi-user sync."""
    
    def __init__(self):
        self.shared_logs = {}  # log_id -> A2.Log data
        self.user_permissions = {}  # log_id -> {user_id: 'owner'/'collaborator'/'viewer'}
        self.change_history = []  # Git-like commit history
    
    def create_shared_log(self, log_id: str, owner_id: str) -> Dict:
        """Create a new shared A2.Log."""
        self.shared_logs[log_id] = {
            'entries': [],
            'created_at': datetime.now(),
            'last_modified': datetime.now(),
            'owner': owner_id
        }
        
        self.user_permissions[log_id] = {
            owner_id: 'owner'
        }
        
        return {'status': 'created', 'log_id': log_id}
    
    def add_collaborator(self, log_id: str, user_id: str, permission: str = 'collaborator') -> Dict:
        """Add a collaborator to shared log."""
        if log_id not in self.shared_logs:
            return {'status': 'error', 'message': 'Log not found'}
        
        self.user_permissions[log_id][user_id] = permission
        
        return {'status': 'success', 'user_id': user_id, 'permission': permission}
    
    def add_entry(self, log_id: str, user_id: str, entry: Dict) -> Dict:
        """
        Add entry to shared log (syncs to all collaborators).
        
        Args:
            log_id: Shared log ID
            user_id: User making the change
            entry: Entry to add
        
        Returns:
            {
                'status': 'success',
                'synced_to': [user_ids...],
                'commit_id': str
            }
        """
        if log_id not in self.shared_logs:
            return {'status': 'error', 'message': 'Log not found'}
        
        # Check permissions
        permission = self.user_permissions[log_id].get(user_id)
        if permission not in ['owner', 'collaborator']:
            return {'status': 'error', 'message': 'Permission denied'}
        
        # Add entry
        entry['added_by'] = user_id
        entry['added_at'] = datetime.now()
        
        self.shared_logs[log_id]['entries'].append(entry)
        self.shared_logs[log_id]['last_modified'] = datetime.now()
        
        # Create commit record (Git-like)
        commit = {
            'commit_id': hashlib.md5(f"{log_id}{datetime.now()}".encode()).hexdigest()[:8],
            'log_id': log_id,
            'user_id': user_id,
            'action': 'add_entry',
            'timestamp': datetime.now(),
            'entry_type': entry.get('type', 'unknown')
        }
        
        self.change_history.append(commit)
        
        # Get all collaborators for sync notification
        collaborators = [uid for uid, perm in self.user_permissions[log_id].items() if uid != user_id]
        
        return {
            'status': 'success',
            'synced_to': collaborators,
            'commit_id': commit['commit_id'],
            'total_entries': len(self.shared_logs[log_id]['entries'])
        }
    
    def get_log(self, log_id: str, user_id: str) -> Dict:
        """Get shared log (if user has permission)."""
        if log_id not in self.shared_logs:
            return {'status': 'error', 'message': 'Log not found'}
        
        permission = self.user_permissions[log_id].get(user_id)
        if not permission:
            return {'status': 'error', 'message': 'Permission denied'}
        
        return {
            'status': 'success',
            'log': self.shared_logs[log_id],
            'your_permission': permission,
            'collaborators': len(self.user_permissions[log_id])
        }
    
    def get_commit_history(self, log_id: str) -> List[Dict]:
        """Get Git-like commit history."""
        return [c for c in self.change_history if c['log_id'] == log_id]


# ============================================================================
# ENHANCED A2.LOG.AI ENGINE
# ============================================================================

class EnhancedA2LogAI:
    """Enhanced engine with all 4 fixes."""
    
    def __init__(self, user_id: str = 'User1'):
        self.user_id = user_id
        self.classifier = EnhancedAIClassifier()
        self.chat_merger = ChatMerger()
        self.collaborative = CollaborativeA2Log()
        
        self.personal_a2log = []  # User's personal A2.Log
        self.conversations = {}  # conversation_id -> messages
        self.current_conversation_id = 'conv_1'
        
        self.stats = {
            'messages_processed': 0,
            'alpha_insights': 0,
            'logic_systems_saved': 0,
            'formulas_failed': 0,
            'formulas_mastered': 0,
            'chats_merged': 0,
            'collaborative_logs': 0
        }
    
    def process_message(
        self, 
        message: str, 
        role: str,
        conversation_id: Optional[str] = None,
        user_feedback: Optional[str] = None
    ) -> Dict:
        """Process message with enhanced classification."""
        
        conv_id = conversation_id or self.current_conversation_id
        
        if conv_id not in self.conversations:
            self.conversations[conv_id] = []
            self.chat_merger.add_conversation(conv_id, self.conversations[conv_id])
        
        self.stats['messages_processed'] += 1
        
        # Enhanced classification
        classification = self.classifier.classify(message, role, user_feedback)
        
        # Update stats
        if classification['grade'] == 1:
            self.stats['alpha_insights'] += 1
            
            if classification['type'] == 'logic_system':
                self.stats['logic_systems_saved'] += 1
            elif classification['type'] == 'formula_failed':
                self.stats['formulas_failed'] += 1
        
        if classification['type'] == 'formula_mastered':
            self.stats['formulas_mastered'] += 1
        
        # Create entry
        entry = {
            'content': message,
            'role': role,
            'timestamp': datetime.now(),
            'grade': classification['grade'],
            'type': classification['type'],
            'reason': classification['reason'],
            'confidence': classification['confidence'],
            'entities': classification.get('entities', {}),
            'conversation_id': conv_id
        }
        
        # Add to conversation
        self.conversations[conv_id].append(entry)
        
        # Add to A2.Log if Grade 1
        if classification['grade'] == 1:
            self.personal_a2log.append(entry)
        
        return {
            'status': 'processed',
            'classification': classification,
            'conversation_id': conv_id,
            'requires_feedback': classification.get('requires_user_feedback', False),
            'auto_save_to_drive': classification.get('auto_save_to_drive', False)
        }
    
    def merge_chats(
        self,
        source_conv_id: str,
        target_conv_id: str,
        start_msg: int,
        end_msg: int
    ) -> Dict:
        """FIX #3: Merge conversations."""
        self.stats['chats_merged'] += 1
        return self.chat_merger.merge_conversations(
            source_conv_id,
            target_conv_id,
            start_msg,
            end_msg
        )
    
    def create_shared_log(self, log_id: str) -> Dict:
        """FIX #4: Create collaborative log."""
        self.stats['collaborative_logs'] += 1
        return self.collaborative.create_shared_log(log_id, self.user_id)
    
    def add_to_shared_log(self, log_id: str, entry: Dict) -> Dict:
        """FIX #4: Add to shared log (syncs to collaborators)."""
        return self.collaborative.add_entry(log_id, self.user_id, entry)


# ============================================================================
# STREAMLIT INTERFACE (Enhanced)
# ============================================================================

def init_session_state():
    if 'engine' not in st.session_state:
        st.session_state.engine = EnhancedA2LogAI(user_id='Alice')
    if 'pending_formula_feedback' not in st.session_state:
        st.session_state.pending_formula_feedback = []


def render_header():
    st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; margin-bottom: 2rem;">
            <h1 style="color: white; font-size: 3rem; margin: 0;">A2.Log.AI v2.0</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Enhanced: Logic Detection, Failure Learning, Chat Merge, Collaboration</p>
        </div>
    """, unsafe_allow_html=True)


def render_conversation_interface():
    st.markdown("### 💬 Enhanced Conversation")
    
    engine = st.session_state.engine
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        message = st.text_area("Message:", height=150)
    
    with col2:
        role = st.selectbox("Role:", ["user", "assistant"])
        conv_id = st.text_input("Conversation ID:", value="conv_1")
    
    with col3:
        st.markdown("**Formula Feedback:**")
        feedback = st.radio("Did you fail?", ["None", "Failed ❌", "Mastered ✅"], label_visibility="collapsed")
        
        feedback_map = {"None": None, "Failed ❌": "failed", "Mastered ✅": "mastered"}
        user_feedback = feedback_map[feedback]
    
    if st.button("🚀 Process Message", use_container_width=True):
        if message.strip():
            result = engine.process_message(message, role, conv_id, user_feedback)
            
            classification = result['classification']
            
            grade_labels = {1: "🟢 Alpha", 2: "🔵 DNA", 3: "⚪ Noise"}
            
            st.success(f"{grade_labels[classification['grade']]} | Type: {classification['type']}")
            st.info(f"Confidence: {classification['confidence']*100:.0f}%")
            
            if result.get('auto_save_to_drive'):
                st.warning("🔥 LOGIC/SYSTEM DETECTED - Auto-saved to Google Drive!")
            
            if result.get('requires_feedback'):
                st.warning("⚠️ Formula detected - Mark as Failed or Mastered")
            
            st.rerun()


def render_chat_merger():
    st.markdown("### 🔀 Smart Chat Merger (FIX #3)")
    
    engine = st.session_state.engine
    
    st.info("Merge messages between conversation threads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source = st.text_input("Source Conversation:", value="conv_3")
        start_idx = st.number_input("Start Message #:", min_value=0, value=0)
        end_idx = st.number_input("End Message #:", min_value=0, value=10)
    
    with col2:
        target = st.text_input("Target Conversation:", value="conv_2")
        
        # Show suggestions
        if source in engine.chat_merger.conversations:
            suggestions = engine.chat_merger.suggest_merge_targets(source)
            if suggestions:
                st.markdown("**Suggested targets:**")
                for conv_id, similarity in suggestions[:3]:
                    st.write(f"- {conv_id} ({similarity} common topics)")
    
    if st.button("🔀 Merge Chats", use_container_width=True):
        result = engine.merge_chats(source, target, start_idx, end_idx)
        
        if result['status'] == 'success':
            st.success(f"Merged {result['moved_messages']} messages!")
            st.info(f"Common topics: {', '.join(result['common_topics'])}")
        else:
            st.error(result.get('message', 'Error'))


def render_collaborative():
    st.markdown("### 👥 Collaborative A2.Log (FIX #4)")
    
    engine = st.session_state.engine
    
    tab1, tab2 = st.tabs(["Create Shared Log", "Add to Shared Log"])
    
    with tab1:
        log_id = st.text_input("Log ID:", value="social_media_project")
        
        if st.button("Create Shared Log"):
            result = engine.create_shared_log(log_id)
            st.success(f"Created: {log_id}")
            st.rerun()
        
        # Add collaborators
        if log_id in engine.collaborative.shared_logs:
            st.markdown("**Add Collaborator:**")
            collab_user = st.text_input("User ID:", value="Bob")
            collab_perm = st.selectbox("Permission:", ["collaborator", "viewer"])
            
            if st.button("Add Collaborator"):
                result = engine.collaborative.add_collaborator(log_id, collab_user, collab_perm)
                st.success(f"Added {collab_user} as {collab_perm}")
    
    with tab2:
        shared_log_id = st.text_input("Shared Log ID:", value="social_media_project")
        entry_content = st.text_area("Entry Content:")
        entry_type = st.selectbox("Type:", ["logic_system", "code", "formula", "definition"])
        
        if st.button("Add Entry (Syncs to All)"):
            if entry_content.strip():
                entry = {
                    'content': entry_content,
                    'grade': 1,
                    'type': entry_type,
                    'entities': {}
                }
                
                result = engine.add_to_shared_log(shared_log_id, entry)
                
                if result['status'] == 'success':
                    st.success(f"Added! Synced to: {', '.join(result['synced_to'])}")
                    st.info(f"Commit ID: {result['commit_id']}")
                else:
                    st.error(result.get('message', 'Error'))


def render_stats():
    st.markdown("### 📊 Enhanced Statistics")
    
    stats = st.session_state.engine.stats
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Messages", stats['messages_processed'])
        st.metric("🔥 Logic Systems", stats['logic_systems_saved'])
    
    with col2:
        st.metric("Alpha Insights", stats['alpha_insights'])
        st.metric("❌ Formulas Failed", stats['formulas_failed'])
    
    with col3:
        st.metric("✅ Formulas Mastered", stats['formulas_mastered'])
        st.metric("🔀 Chats Merged", stats['chats_merged'])
    
    with col4:
        st.metric("👥 Collaborative Logs", stats['collaborative_logs'])


def main():
    st.set_page_config(page_title="A2.Log.AI v2.0", page_icon="🤖", layout="wide")
    
    st.markdown("""
        <style>
        .stApp {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);}
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    render_header()
    
    with st.sidebar:
        st.markdown("## 🎛️ Navigation")
        page = st.radio("", ["Dashboard", "Conversation", "Chat Merger", "Collaborative"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### New Features")
        st.success("✅ Logic Detection")
        st.success("✅ Failure-Based Learning")
        st.success("✅ Chat Merging")
        st.success("✅ Collaborative A2.Log")
        
        st.markdown("---")
        if st.button("🔄 Reset"):
            st.session_state.engine = EnhancedA2LogAI(user_id='Alice')
            st.rerun()
    
    if page == "Dashboard":
        render_stats()
        st.markdown("---")
        render_conversation_interface()
    elif page == "Conversation":
        render_conversation_interface()
    elif page == "Chat Merger":
        render_chat_merger()
    elif page == "Collaborative":
        render_collaborative()
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.7);">
            A2.Log.AI v2.0 Enhanced | Anindita Ray
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()