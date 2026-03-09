import graphviz

def create_architecture_diagram():
    # Initialize the directed graph
    dot = graphviz.Digraph('Chronic_Disease_Chatbot', comment='System Architecture Dataflow')
    
    # Graph global attributes for a clean, modern look
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.0')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Helvetica', fontsize='12', margin='0.2')
    dot.attr('edge', fontname='Helvetica', fontsize='10', color='#666666')

    # ==========================================
    # Layer 1: Presentation Layer
    # ==========================================
    with dot.subgraph(name='cluster_presentation') as c:
        c.attr(label='Layer 1: Presentation Layer', style='dashed', color='#aaaaaa', fontname='Helvetica-Bold')
        c.node('UI', 'React / Streamlit UI\n(Browser / Frontend)', fillcolor='#e3f2fd')

    # ==========================================
    # Layer 2: API Gateway Layer
    # ==========================================
    with dot.subgraph(name='cluster_api') as c:
        c.attr(label='Layer 2: API Gateway Layer', style='dashed', color='#aaaaaa', fontname='Helvetica-Bold')
        c.node('FastAPI', 'FastAPI Server\n(Session Management & Validation)', fillcolor='#e8f5e9')

    # ==========================================
    # Layer 3: Orchestration Layer
    # ==========================================
    with dot.subgraph(name='cluster_orchestration') as c:
        c.attr(label='Layer 3: Orchestration Layer (LangGraph)', style='dashed', color='#aaaaaa', fontname='Helvetica-Bold')
        
        # Central Orchestrator
        c.node('Orchestrator', 'Main Agent (Orchestrator)\nGemini 1.5 Pro', fillcolor='#fff3e0', shape='octagon', style='filled,bold')
        
        # Sub-agents grouped together
        with dot.subgraph(name='cluster_agents') as agents:
            agents.attr(style='invis') # Invisible boundary just for alignment
            agents.node('Knowledge', 'Knowledge Agent', fillcolor='#f3e5f5')
            agents.node('Memory', 'Memory Agent', fillcolor='#f3e5f5')
            agents.node('Action', 'Action Agent', fillcolor='#f3e5f5')

    # ==========================================
    # Layers 4 & 5: Data, External Services & Persistence
    # ==========================================
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='Layers 4 & 5: Data, Services & Persistence', style='dashed', color='#aaaaaa', fontname='Helvetica-Bold')
        
        # External APIs
        c.node('Tavily', 'Tavily API\n(Web Search)', fillcolor='#ffebee', shape='component')
        c.node('Calendar', 'Google Calendar API\n(Booking)', fillcolor='#ffebee', shape='component')
        c.node('SendGrid', 'SendGrid API\n(Email)', fillcolor='#ffebee', shape='component')
        
        # Persistent Storage
        c.node('ChromaDB', 'ChromaDB\n(Vector Symptom Logs)', fillcolor='#e0f7fa', shape='cylinder')
        c.node('SQLite', 'SQLite\n(Structured Patient Data)', fillcolor='#e0f7fa', shape='cylinder')

    # ==========================================
    # Define Data Flow (Edges)
    # ==========================================
    
    # User to Server
    dot.edge('UI', 'FastAPI', label=' HTTP POST /chat\n(JSON Body)')
    
    # Server to LangGraph
    dot.edge('FastAPI', 'Orchestrator', label=' invokes app with\nAgentState')
    
    # Orchestrator routing loops (Bidirectional)
    dot.edge('Orchestrator', 'Knowledge', label=' routes task\nreturns context', dir='both', color='#ff9800', penwidth='1.5')
    dot.edge('Orchestrator', 'Memory', label=' routes task\nreturns context', dir='both', color='#ff9800', penwidth='1.5')
    dot.edge('Orchestrator', 'Action', label=' routes task\nreturns status', dir='both', color='#ff9800', penwidth='1.5')

    # Agents to Storage/Services
    dot.edge('Knowledge', 'Tavily', label=' Outbound GET')
    dot.edge('Memory', 'ChromaDB', label=' Semantic Search /\nWrite Vector', dir='both')
    dot.edge('Memory', 'SQLite', label=' SQL Read/Write', dir='both')
    dot.edge('Action', 'Calendar', label=' Outbound POST')
    dot.edge('Action', 'SendGrid', label=' Outbound SMTP')

    # Render the diagram to a file
    output_filename = 'chronic_disease_chatbot_architecture'
    dot.render(output_filename, format='png', cleanup=True)
    print(f"Diagram successfully generated as {output_filename}.png")

if __name__ == '__main__':
    create_architecture_diagram()