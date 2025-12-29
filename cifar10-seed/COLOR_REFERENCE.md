# CIFAR-10 Federated Learning Methods - Color Reference

## 🎨 **Color Mapping for All Methods**

Use these exact hex codes for consistent coloring across all plots and visualizations:

| Method | Display Name | Hex Color | RGB Values | Color Description |
|--------|-------------|-----------|------------|-------------------|
| **Scaffold** | Scaffold | `#1f77b4` | (31, 119, 180) | Blue |
| **FedProx** | FedProx | `#ff7f0e` | (255, 127, 14) | Orange |
| **pFedMe** | pFedMe | `#2ca02c` | (44, 160, 44) | Green |
| **HierFL** | HierFL | `#d62728` | (214, 39, 40) | Red |
| **CFL** | CFL | `#9467bd` | (148, 103, 189) | Purple |
| **FEDGE** | Fedge-v1 | `#8c564b` | (140, 86, 75) | Brown |

## 📊 **Legend Order**

The methods appear in the legend in this specific order (Fedge-v1 at the end):
1. Scaffold (Blue)
2. FedProx (Orange) 
3. pFedMe (Green)
4. HierFL (Red)
5. CFL (Purple)
6. **Fedge-v1 (Brown)** ← Always last

## 💻 **Code Usage**

### Python/Matplotlib
```python
method_colors = {
    'SCAFFOLD': '#1f77b4',    # Blue
    'FEDPROX': '#ff7f0e',     # Orange  
    'PFEDME': '#2ca02c',      # Green
    'HIERFL': '#d62728',      # Red
    'CFL': '#9467bd',         # Purple
    'FEDGE-V1': '#8c564b'     # Brown
}

# Plot in this order for consistent legend
method_order = ['scaffold', 'fedprox', 'pfedme', 'hierfl', 'cfl', 'fedge']
```

### CSS/Web
```css
.scaffold { color: #1f77b4; }
.fedprox { color: #ff7f0e; }
.pfedme { color: #2ca02c; }
.hierfl { color: #d62728; }
.cfl { color: #9467bd; }
.fedge-v1 { color: #8c564b; }
```

### LaTeX/TikZ
```latex
\definecolor{scaffold}{HTML}{1f77b4}
\definecolor{fedprox}{HTML}{ff7f0e}
\definecolor{pfedme}{HTML}{2ca02c}
\definecolor{hierfl}{HTML}{d62728}
\definecolor{cfl}{HTML}{9467bd}
\definecolor{fedgev1}{HTML}{8c564b}
```

## 🎯 **Performance Ranking (for reference)**

When showing results, use this performance-based ranking:

1. **🥇 Fedge-v1** (Brown) - 85.43% ± 0.00%
2. **🥈 Scaffold** (Blue) - 60.91% ± 2.81%
3. **🥉 FedProx** (Orange) - 59.19% ± 1.30%
4. **CFL** (Purple) - 56.11% ± 0.42%
5. **HierFL** (Red) - 44.31% ± 0.51%
6. **pFedMe** (Green) - 10.02% ± 0.31%

## 📝 **Notes**

- Colors are based on matplotlib's default color cycle for consistency
- Fedge-v1 uses brown (`#8c564b`) to distinguish it as the newest/best method
- Legend order puts Fedge-v1 last to emphasize it as the final/best solution
- All colors have sufficient contrast for accessibility
- Colors work well in both light and dark backgrounds
