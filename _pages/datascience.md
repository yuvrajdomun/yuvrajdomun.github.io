---
title: "Blogs"
layout: single
permalink: /blogsabsolute_url`/
---


<!-- Highlights -->

{% if paginator %}
{% assign posts = paginator.posts %}
{% else %}
{% assign posts = site.posts %}
{% endif %}

{% assign entries_layout = page.entries_layout | default: 'section' %}

{% for post in posts %}

        {% if page.header.overlay_color or page.header.overlay_image or page.header.image %}
        {% include page__hero.html %}
        {% elsif page.header.video.id and page.header.video.provider %}
            {% include page__hero_video.html %}
        {% endif %}
        {% include archive-single.html type=entries_layout %}

{% endfor %}


