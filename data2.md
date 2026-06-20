\begin{table*}[!htp]
\centering
\begin{threeparttable}

\caption{Segmentation performance across multi thyroid ultrasound datasets. Upper block, Dice score; lower block, 95th percentile Hausdorff distance (HD95). Values are reported with 95\% confidence intervals.}
\label{tab:seg_performance2}

\footnotesize
\setlength{\tabcolsep}{5.5pt}
\renewcommand{\arraystretch}{1.08}

\begin{tabular}{lcccccc}
\toprule
\textbf{Model}
& \textbf{TN3K}
& \textbf{ThyroidXL}
& \textbf{PKTN}
& \textbf{TN5K}
& \textbf{DDTI}
& \textbf{Augtrain} \\
\midrule
TransUnet~\cite{chen2024transunet}
& $81.84 \pm 1.62$
& $85.75 \pm 0.57$
& $76.89 \pm 3.56$
& $78.54 \pm 1.51$
& $76.96 \pm 4.79$
& --- \\
MedSegX~\cite{zhang2025generalist}
& $83.93 \pm 0.79$
& $79.98 \pm 0.36$
& $80.63 \pm 0.42$
& $83.10 \pm 0.48$
& $85.34 \pm 0.42$
& --- \\
MedSAM2~\cite{ma2025medsam2}
& $84.47 \pm 1.02$
& $83.74 \pm 0.46$
& $80.71 \pm 0.98$
& $81.22 \pm 1.14$
& $84.90 \pm 1.51$
& --- \\
UltraFedFM~\cite{jiang2025pretraining}
& $81.18 \pm 1.46$
& $84.70 \pm 0.53$
& $75.31 \pm 1.12$
& $77.13 \pm 1.38$
& $75.55 \pm 1.57$
& --- \\
\rowcolor{lightgray}
\textbf{ThyroidXAgent}
& $85.28 \pm 1.28$
& $87.58 \pm 0.44$
& $82.96 \pm 1.98$
& $83.26 \pm 1.34$
& $85.16 \pm 2.71$
& --- \\

\midrule
TransUnet~\cite{chen2024transunet}
& $27.27 \pm 5.52$
& $22.42 \pm 1.34$
& $26.88 \pm 9.66$
& $22.32 \pm 3.43$
& $20.46 \pm 5.54$
& --- \\
MedSegX~\cite{zhang2025generalist}
& $10.95 \pm 0.64$
& $11.07 \pm 0.32$
& $10.83 \pm 0.70$
& $11.76 \pm 0.76$
& $10.44 \pm 0.48$
& --- \\
MedSAM2~\cite{ma2025medsam2}
& $11.51 \pm 1.53$
& $6.91 \pm 0.44$
& $10.69 \pm 2.34$
& $10.94 \pm 1.12$
& $9.68 \pm 2.08$
& --- \\
UltraFedFM~\cite{jiang2025pretraining}
& $14.98 \pm 2.10$
& $8.10 \pm 0.58$
& $16.08 \pm 1.67$
& $14.96 \pm 1.65$
& $18.10 \pm 1.52$
& --- \\
\rowcolor{lightgray}
\textbf{ThyroidXAgent}
& $10.31 \pm 1.70$
& $5.43 \pm 0.53$
& $9.01 \pm 3.58$
& $10.12 \pm 1.23$
& $9.44 \pm 3.54$
& --- \\

\bottomrule
\end{tabular}
\end{threeparttable}
\end{table*}