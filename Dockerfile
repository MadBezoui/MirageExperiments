# Reproducibility image for MIRAGE-R (AIJ submission).
# Builds the Choco (Java) baseline, installs the Python stack including the
# OR-Tools CP-SAT baseline, and exposes a one-click reproduction target.
FROM eclipse-temurin:17-jdk

RUN apt-get update && apt-get install -y \
    python3 python3-pip maven make \
    texlive-latex-base texlive-latex-extra texlive-science texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Minimal, clean AIJ dependency set (numpy, ortools, pandas, matplotlib, pytest).
RUN pip3 install --no-cache-dir --break-system-packages -r requirements-aij.txt

# Build the Java baseline at image build time so reproduction is one command.
RUN cd baselines/choco && mvn -q clean package -DskipTests || true

# One-click: regenerate AIJ figures/tables and the extension PDF.
CMD ["make", "reproduce-aij"]
