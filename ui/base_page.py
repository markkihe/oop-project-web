"""คลาสแม่ของทุกหน้าในแอป"""
from abc import ABC, abstractmethod

import streamlit as st


class BasePage(ABC):
    """Template Method: run() กำหนดลำดับการแสดงผลเหมือนกันทุกหน้า
    คลาสลูกเขียนทับเฉพาะส่วนที่ต่างกัน (โดยเฉพาะ render_body)"""

    def __init__(self, title: str, icon: str, page_title: str = None, layout: str = "wide"):
        self._title = title
        self._icon = icon
        self._page_title = page_title or title
        self._layout = layout

    def run(self):
        st.set_page_config(page_title=self._page_title, page_icon=self._icon, layout=self._layout)
        self.render_header()
        self.render_body()
        self.render_footer()

    def render_header(self):
        st.title(f"{self._icon} {self._title}")
        st.markdown("---")

    @abstractmethod
    def render_body(self):
        ...

    def render_footer(self):
        pass


class Section:
    """หัวข้อ 1 ส่วนในหน้าอธิบาย"""

    def __init__(self, heading: str, content: str):
        self.heading = heading
        self.content = content


class SectionPage(BasePage):
    """หน้าที่ประกอบด้วยหลายหัวข้อ (Composition กับ Section) ใช้กับหน้าอธิบายต่าง ๆ"""

    def __init__(self, title, icon, sections, page_title=None, footer_message=None,
                 divider_between_sections=False):
        super().__init__(title, icon, page_title)
        self._sections = list(sections)
        self._footer_message = footer_message
        self._divider = divider_between_sections

    def render_body(self):
        for i, section in enumerate(self._sections):
            if self._divider and i > 0:
                st.markdown("---")
            st.header(section.heading)
            st.markdown(section.content)

    def render_footer(self):
        if self._footer_message:
            if self._divider:
                st.markdown("---")
            st.success(self._footer_message)
