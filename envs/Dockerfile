FROM nvcr.io/nvidia/pytorch:23.05-py3

ARG user="ksterx" \
    email="kostonerx@gmail.com"

LABEL maintainer="${user} <${email}>"

ENV TZ=Asia/Tokyo \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    htop \
    python3-tk \
    zsh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && chsh -s $(which zsh) \
    && sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && sed -i -e 's/plugins=(git)/plugins=(git zsh-autosuggestions)/g' ~/.zshrc \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && echo "set completion-ignore-case on" >> ${HOME}/.inputrc \
    && git config --global user.email ${email} \
    && git config --global user.name ${user}
