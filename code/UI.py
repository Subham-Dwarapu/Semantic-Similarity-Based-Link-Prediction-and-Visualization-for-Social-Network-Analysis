import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import predict_links_gcn, predict_links_graphsage, predict_links_gat
from predict_links_gcn import main
from predict_links_graphsage import main
from predict_links_gat import main
import os
from tkinter.font import Font

class ModernUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Link Prediction Tool")
        self.root.geometry("850x650")
        self.root.configure(bg="#f8f9fa")
        
        # Set minimum window size
        self.root.minsize(800, 600)
        
        # Initialize variables first
        self.gml_path = ""
        self.model_path = ""
        self.status_message = tk.StringVar()
        self.status_message.set("Ready to predict links")
        
        # Create custom fonts
        self.header_font = ('Segoe UI', 16, 'bold')
        self.subheader_font = ('Segoe UI', 12, 'bold')
        self.normal_font = ('Segoe UI', 10)
        self.small_font = ('Segoe UI', 9)
        
        # Configure application styles
        self.configure_styles()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Initialize rest of UI
        self.create_header()
        self.create_content()
        self.create_status_bar()

    def configure_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Define colors
        bg_color = "#f8f9fa"
        frame_bg = "#ffffff"
        accent_color = "#4a6ea9"
        
        # Configure styles
        self.style.configure("Main.TFrame", background=bg_color)
        self.style.configure("Header.TLabel", 
                           background=bg_color,
                           font=self.header_font)
        self.style.configure("TLabel",
                           background=frame_bg,
                           font=self.normal_font)
        self.style.configure("Status.TLabel",
                           background="#f1f3f4",
                           font=self.small_font)

    def create_header(self):
        header_frame = ttk.Frame(self.main_frame, style="Main.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header_frame, 
                 text="Link Prediction Tool",
                 style="Header.TLabel").pack(side=tk.LEFT)

    def create_content(self):
        # Create content area
        self.content_frame = ttk.Frame(self.main_frame, style="Main.TFrame")
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Input section
        self.create_input_section()
        
        # Results section
        self.create_results_section()

    def create_input_section(self):
        # Create input frame (card-like appearance)
        self.input_frame = ttk.LabelFrame(self.content_frame, 
                                          text="Input Parameters", 
                                          style="Card.TLabelframe")
        self.input_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        # Add padding inside the frame
        inner_frame = ttk.Frame(self.input_frame, style="Card.TFrame")
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # File selection section
        file_frame = ttk.Frame(inner_frame, style="Card.TFrame")
        file_frame.pack(fill=tk.X, expand=False)
        
        # GML File row
        gml_frame = ttk.Frame(file_frame, style="Card.TFrame")
        gml_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(gml_frame, text="Network File (GML):", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        self.gml_entry = ttk.Entry(gml_frame, width=50)
        self.gml_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(gml_frame, 
                   text="Browse...", 
                   command=self.select_gml_file, 
                   style="File.TButton").pack(side=tk.LEFT, padx=5)
        
        # Model File row
        model_frame = ttk.Frame(file_frame, style="Card.TFrame")
        model_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(model_frame, text="Model File (PT):", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        self.model_entry = ttk.Entry(model_frame, width=50)
        self.model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(model_frame, 
                   text="Browse...", 
                   command=self.select_model_file, 
                   style="File.TButton").pack(side=tk.LEFT, padx=5)
        
        # Add Model Type Selection
        model_type_frame = ttk.Frame(file_frame, style="Card.TFrame")
        model_type_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(model_type_frame, text="Model Type:", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        self.model_type = tk.StringVar(value="GCN")
        model_dropdown = ttk.Combobox(model_type_frame, 
                                    textvariable=self.model_type,
                                    values=["GCN", "GAT", "GraphSage"],
                                    state="readonly",
                                    width=15)
        model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Parameter section
        param_frame = ttk.Frame(inner_frame, style="Card.TFrame")
        param_frame.pack(fill=tk.X, expand=False, pady=10)
        
        # Create two columns
        left_params = ttk.Frame(param_frame, style="Card.TFrame")
        left_params.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_params = ttk.Frame(param_frame, style="Card.TFrame")
        right_params.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Top K parameter
        top_k_frame = ttk.Frame(left_params, style="Card.TFrame")
        top_k_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(top_k_frame, text="Top K Links:", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        self.top_k_entry = ttk.Entry(top_k_frame, width=10)
        self.top_k_entry.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(top_k_frame, 
                  text="(Leave empty for all links)", 
                  style="Status.TLabel").pack(side=tk.LEFT)
        
        # Threshold parameter
        threshold_frame = ttk.Frame(right_params, style="Card.TFrame")
        threshold_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(threshold_frame, text="Threshold:", style="TLabel").pack(side=tk.LEFT, padx=5)
        
        self.threshold_entry = ttk.Entry(threshold_frame, width=10)
        self.threshold_entry.insert(0, "0.5029")
        self.threshold_entry.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(threshold_frame, 
                  text="(Default: 0.5029)", 
                  style="Status.TLabel").pack(side=tk.LEFT)
        
        # Run button in its own frame
        button_frame = ttk.Frame(inner_frame, style="Card.TFrame")
        button_frame.pack(fill=tk.X, expand=False, pady=(15, 5))
        
        self.run_button = ttk.Button(button_frame, 
                                    text="Run Prediction", 
                                    command=self.run_prediction, 
                                    style="Accent.TButton")
        self.run_button.pack(side=tk.RIGHT, padx=5)

    def create_results_section(self):
        # Create results frame (card-like appearance)
        self.results_frame = ttk.LabelFrame(self.content_frame, 
                                           text="Results", 
                                           style="Card.TLabelframe")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(self.results_frame, style="Card.TFrame")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.result_text = tk.Text(text_frame, 
                                  wrap=tk.WORD, 
                                  bg="#fafbfc", 
                                  fg="#24292e", 
                                  font=self.normal_font,
                                  bd=1, 
                                  relief=tk.SOLID,
                                  padx=10,
                                  pady=10)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        # Set initial text
        self.result_text.insert(tk.END, "Welcome to Link Prediction Tool\n")
        self.result_text.insert(tk.END, "—" * 40 + "\n")
        self.result_text.insert(tk.END, "1. Select a GML network file\n")
        self.result_text.insert(tk.END, "2. Choose model type (GCN, GAT, or GraphSage)\n")
        self.result_text.insert(tk.END, "3. Select a trained model file\n")
        self.result_text.insert(tk.END, "4. Adjust parameters if needed\n")
        self.result_text.insert(tk.END, "5. Click 'Run Prediction'\n")
        self.result_text.configure(state="disabled")

    def create_status_bar(self):
        # Create status bar
        self.status_bar = ttk.Frame(self.root, style="Main.TFrame")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add status separator
        separator = ttk.Separator(self.status_bar, orient="horizontal")
        separator.pack(side=tk.TOP, fill=tk.X)
        
        # Status label
        status_label = ttk.Label(self.status_bar, 
                                textvariable=self.status_message, 
                                style="Status.TLabel")
        status_label.pack(side=tk.LEFT, padx=10, pady=5)

    def select_gml_file(self):
        filepath = filedialog.askopenfilename(
            title="Select GML Network File", 
            filetypes=[("GML Files", "*.gml"), ("All Files", "*.*")]
        )
        if filepath:
            self.gml_path = filepath
            self.gml_entry.delete(0, tk.END)
            self.gml_entry.insert(0, filepath)
            self.status_message.set(f"Selected network file: {os.path.basename(filepath)}")

    def select_model_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Model File", 
            filetypes=[("PT Files", "*.pt"), ("All Files", "*.*")]
        )
        if filepath:
            self.model_path = filepath
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, filepath)
            self.status_message.set(f"Selected model file: {os.path.basename(filepath)}")

    def update_results(self, message, is_error=False):
        self.result_text.configure(state="normal")
        self.result_text.delete(1.0, tk.END)
        
        if is_error:
            self.result_text.insert(tk.END, "❌ Error: " + message + "\n", "error")
        else:
            self.result_text.insert(tk.END, message)
        
        self.result_text.configure(state="disabled")
        self.result_text.see(tk.END)

    def run_prediction(self):
        try:
            # Get input values
            gml_file = self.gml_entry.get()
            model_file = self.model_entry.get()
            model_type = self.model_type.get()
            top_k = self.top_k_entry.get()
            threshold = self.threshold_entry.get()
            
            # Validate inputs
            if not gml_file or not model_file:
                messagebox.showerror("Missing Input", "Please select both a network file and model file.")
                return
            
            # Convert inputs
            top_k = int(top_k) if top_k else None
            threshold = float(threshold) if threshold else 0.5029
            
            # Show progress
            self.status_message.set("Processing... Please wait")
            self.update_results(f"⏳ Processing network graph using {model_type}...\n\n"
                              "This may take a moment depending on the size of your network.")
            self.root.update()
            
            # Select appropriate predictor based on model type
            if model_type == "GCN":
                enhanced_graph, predictions = predict_links_gcn.main(
                    gml_file, model_file, top_k, threshold
                )
            elif model_type == "GAT":
                enhanced_graph, predictions = predict_links_gat.main(
                    gml_file, model_file, top_k, threshold
                )
            else:  # GraphSage
                enhanced_graph, predictions = predict_links_graphsage.main(
                    gml_file, model_file, top_k, threshold
                )
            
            # Format results
            result_message = f"""✅ Prediction completed successfully using {model_type}!
            
Network: {os.path.basename(gml_file)}
Model: {os.path.basename(model_file)}
            
📊 Results Summary:
• Model Type: {model_type}
• Predicted {len(predictions)} new edges
• Output files saved with '_{model_type.lower()}_predictions' suffix
• Threshold value: {threshold}
            
The enhanced graph has been saved and is ready for further analysis.
            """
            
            # Update UI
            self.update_results(result_message)
            self.status_message.set(f"Completed: {len(predictions)} links predicted using {model_type}")
            
        except Exception as e:
            self.update_results(str(e), True)
            self.status_message.set(f"Error occurred during {model_type} prediction")

# Create and run the application
def start_ui():
    root = tk.Tk()
    app = ModernUI(root)
    root.mainloop()

if __name__ == "__main__":
    start_ui()