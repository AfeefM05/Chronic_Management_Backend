import graphviz

def create_er_diagram():
    # Initialize the directed graph with a Left-to-Right layout
    er = graphviz.Digraph('Chronic_Disease_ERD', filename='chronic_disease_erd', format='png')
    er.attr(rankdir='LR', nodesep='0.5', ranksep='1.0')
    er.attr('node', shape='none', fontname='Helvetica', fontsize='10')
    er.attr('edge', fontname='Helvetica', fontsize='9', color='#555555')

    # Helper function to generate HTML-like table nodes for Graphviz
    def make_table(name, columns, bgcolor="#f8f9fa"):
        rows = ""
        for col in columns:
            # Highlight Primary Keys (PK) and Foreign Keys (FK)
            if "PK" in col:
                rows += f'<TR><TD ALIGN="LEFT" PORT="{col.split()[0]}"><B>🔑 {col}</B></TD></TR>'
            elif "FK" in col:
                rows += f'<TR><TD ALIGN="LEFT" PORT="{col.split()[0]}"><I>🔗 {col}</I></TD></TR>'
            else:
                rows += f'<TR><TD ALIGN="LEFT" PORT="{col.split()[0]}">{col}</TD></TR>'
                
        return f'''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
            <TR><TD BGCOLOR="{bgcolor}"><B>{name}</B></TD></TR>
            {rows}
        </TABLE>
        >'''

    # ==========================================
    # 1. Define SQLite Tables (Relational)
    # ==========================================
    
    er.node('USER_PROFILE', make_table('USER_PROFILE', [
        'key (PK, TEXT)', 'value (TEXT)'
    ], bgcolor="#e3f2fd"))

    er.node('CONDITIONS', make_table('CONDITIONS', [
        'id (PK, INT)', 'name (TEXT)', 'category (TEXT)', 
        'severity (TEXT)', 'icd_code (TEXT)', 'diagnosed_at (DAT)'
    ], bgcolor="#e8f5e9"))

    er.node('DOCTORS', make_table('DOCTORS', [
        'id (PK, INT)', 'name (TEXT)', 'specialty (TEXT)', 
        'email (TEXT)', 'phone (TEXT)'
    ], bgcolor="#fff3e0"))

    er.node('MEDICATIONS', make_table('MEDICATIONS', [
        'id (PK, INT)', 'condition_id (FK, INT)', 'name (TEXT)', 
        'dose (TEXT)', 'frequency (TEXT)', 'is_active (BOOL)', 'stock_count (INT)'
    ], bgcolor="#f3e5f5"))

    er.node('MEDICATION_LOGS', make_table('MEDICATION_LOGS', [
        'id (PK, INT)', 'medication_id (FK, INT)', 'taken_at (DAT)', 
        'was_taken (BOOL)', 'skipped_reason (TEXT)'
    ], bgcolor="#f3e5f5"))

    er.node('APPOINTMENTS', make_table('APPOINTMENTS', [
        'id (PK, INT)', 'doctor_id (FK, INT)', 'condition_id (FK, INT)', 
        'scheduled_at (DAT)', 'status (TEXT)', 'google_calendar_id (TEXT)'
    ], bgcolor="#ffebee"))

    er.node('SYMPTOM_LOGS', make_table('SYMPTOM_LOGS (SQLite)', [
        'id (PK, INT)', 'chroma_doc_id (TEXT)', 'appointment_id (FK, INT)', 
        'description (TEXT)', 'severity (TEXT)', 'occurred_at (DAT)'
    ], bgcolor="#e0f7fa"))

    # Junction Tables
    er.node('DOCTOR_CONDITIONS', make_table('DOCTOR_CONDITIONS', [
        'doctor_id (FK, INT)', 'condition_id (FK, INT)'
    ], bgcolor="#eceff1"))

    er.node('MEDICATION_DOCTORS', make_table('MEDICATION_DOCTORS', [
        'medication_id (FK, INT)', 'doctor_id (FK, INT)'
    ], bgcolor="#eceff1"))

    # ==========================================
    # 2. Define ChromaDB Collection (Vector)
    # ==========================================
    
    er.node('CHROMA_DB', make_table('SYMPTOM_HISTORY (ChromaDB)', [
        'id (UUID)', 'document (TEXT)', 'embedding (VECTOR[384])', 
        'metadata (JSON: date, severity, sqlite_id)'
    ], bgcolor="#fff8e1"))

    # ==========================================
    # 3. Define Relationships (Edges)
    # ==========================================
    
    # Using 'dir="both"' and arrows to show One-to-Many (1:N) relationships
    # The arrow points to the "Many" side.
    
    # Core Flow
    er.edge('USER_PROFILE', 'CONDITIONS', label=' has (1:N)', arrowhead='crow')
    er.edge('CONDITIONS:id', 'MEDICATIONS:condition_id', label=' treated by (1:N)', arrowhead='crow')
    er.edge('CONDITIONS:id', 'APPOINTMENTS:condition_id', label=' scheduled for (1:N)', arrowhead='crow')
    er.edge('DOCTORS:id', 'APPOINTMENTS:doctor_id', label=' attends (1:N)', arrowhead='crow')
    
    # Logs and Tracking
    er.edge('MEDICATIONS:id', 'MEDICATION_LOGS:medication_id', label=' tracked via (1:N)', arrowhead='crow')
    er.edge('APPOINTMENTS:id', 'SYMPTOM_LOGS:appointment_id', label=' links to (1:N)', arrowhead='crow')
    
    # Many-to-Many Junctions
    er.edge('DOCTORS:id', 'DOCTOR_CONDITIONS:doctor_id', arrowhead='crow')
    er.edge('CONDITIONS:id', 'DOCTOR_CONDITIONS:condition_id', arrowhead='crow')
    er.edge('MEDICATIONS:id', 'MEDICATION_DOCTORS:medication_id', arrowhead='crow')
    er.edge('DOCTORS:id', 'MEDICATION_DOCTORS:doctor_id', arrowhead='crow')

    # The SQLite <-> ChromaDB Bridge
    er.edge('SYMPTOM_LOGS:chroma_doc_id', 'CHROMA_DB:id', 
            label=' Bridge (1:1)', arrowhead='none', arrowtail='none', 
            style='dashed', color='#e65100', penwidth='2.0')

    # Render the diagram
    er.render(cleanup=True)
    print("ER Diagram successfully generated as 'chronic_disease_erd.png'")

if __name__ == '__main__':
    create_er_diagram()